#include "stdafx.h"
#include <string>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <time.h>
#include <algorithm>
#include <sstream>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
#define _USE_MATH_DEFINES
#include <math.h>

#define AOCL_ALIGNMENT 64
#define checkError(status, msg) _checkError(__LINE__, __FILE__, status, msg)

//typedef struct {
//	cl_uint width;
//	cl_uint height;
//	cl_uint step;
//	cl_uint filter;
//} ResponseLayer;

struct IPoint_small {
	cl_float x;
	cl_float y;
	cl_float scale;
	cl_uint laplacian;
};

struct IPoint {
	float x;
	float y;
	float scale;
	float orientation;
	unsigned int laplacian;
	int descriptor_length;
	float descriptors[64];
};

//! lookup table for 2d gaussian (sigma = 2.5) where (0,0) is top left and (6,6) is bottom right
const float gauss25[7][7] = {
	{ 0.02546481f,0.02350698f,	0.01849125f,	0.01239505f,	0.00708017f,	0.00344629f,	0.00142946f },
	{ 0.02350698f,	0.02169968f,	0.01706957f,	0.01144208f,	0.00653582f,	0.00318132f,	0.00131956f },
	{ 0.01849125f,	0.01706957f,	0.01342740f,	0.00900066f,	0.00514126f,	0.00250252f,	0.00103800f },
	{ 0.01239505f,	0.01144208f,	0.00900066f,	0.00603332f,	0.00344629f,	0.00167749f,	0.00069579f },
	{ 0.00708017f,	0.00653582f,	0.00514126f,	0.00344629f,	0.00196855f,	0.00095820f,	0.00039744f },
	{ 0.00344629f,	0.00318132f,	0.00250252f,	0.00167749f,	0.00095820f,	0.00046640f,	0.00019346f },
	{ 0.00142946f,	0.00131956f,	0.00103800f,	0.00069579f,	0.00039744f,	0.00019346f,	0.00008024f }
};

// hessian properties
const int OCTAVES = 5;
const int INTERVALS = 4;
const int INITSAMPLE = 2;
const cl_uint filter_map[OCTAVES][INTERVALS] = { { 0,1,2,3 },{ 1,3,4,5 },{ 3,5,6,7 },{ 5,7,8,9 },{ 7,9,10,11 } };

// Arguments
int PLATFORM_NUM;
int DEVICE_NUM;
int FILE_COUNT;
bool PRINT_IPOINTS;

//Raw image
cl_mem rawImage;
cl_uint rawImageWidth;
cl_uint rawImageHeight;

// Platform stuff
cl_platform_id *platforms;
cl_platform_id platformID;
cl_uint platformCount;
cl_device_id devices[2];
cl_device_id deviceID;
cl_command_queue commandQueue;
cl_context context;
cl_int error;
size_t workerCount;

size_t origin[3] = { 0, 0, 0 };
size_t region[3] = { 10, 10, 1 };

// Integral Image
cl_kernel integralKernel;
size_t integralKernelWorkSize;
cl_mem d_inputImage;
cl_mem d_integralImage;
cl_program integralProgram;

float *h_integralImage;

// Hessian response kernel
cl_int h_filterWidth[12];
cl_int h_filterBorder[12];
cl_int h_filterLobe[12];
cl_int h_filterStep[12];
cl_float h_normaliseFactor[12];

cl_mem d_responseMap;
cl_mem d_filterWidth;
cl_mem d_filterBorder;
cl_mem d_filterLobe;
cl_mem d_filterStep;
cl_mem d_normaliseFactor;
cl_mem d_responseBuffer;

cl_image_desc desc2DArray;
cl_mem d_laplacianBuffer;

cl_float *h_responseBuffer;
cl_uint *h_laplacianBuffer;

//ResponseLayer *h_responseMap;
cl_int4 *h_responseMap;

cl_program hessianProgram;
cl_kernel responseKernel;
size_t responseKernelWorkSize;

// Get IPoints Kernel
cl_kernel ipointsKernel;
size_t ipointsKernelGlobalSize;
size_t ipointsKernelLocalSize;
IPoint_small *h_ipts;
cl_uint h_iptsCount;
cl_mem d_ipts;
cl_mem d_iptsIndex;
IPoint *h_iptsDesc;
IPoint_small *h_iptsTempCopy;

// Desc kernel
cl_program descProgram;
cl_kernel descKernel;

int descIptIndex = 0;

inline void _checkError(int line, const char *file, cl_int error, const char *msg); // does not return
void getOrientation(IPoint *ipt);
void getDescriptor(IPoint *ipt);
inline void init();
inline void cleanup();
inline void createIntegralImage(unsigned int width, unsigned int height, float *imageData, float *result);
inline void printPlatformInfo(cl_platform_id platform);
inline void printDeviceInfo(cl_device_id device);
inline float* loadImage(const char *fileName);
inline void createProgram(char  *filepath, cl_program &program);
inline void createKernel(char *kernelName, cl_program &program, cl_kernel &kernel);
inline void initResponseMap();
inline void buildFilterProperties(cl_int4 *responseMap, cl_int *filterWidth, cl_int *filterBorder, cl_int *filterLobe, cl_int *filterStep, cl_float *normaliseFactor);
inline const char *getErrorString(cl_int error);

int main(int argc, char *argv[]) {
	// Get platform count
	/*printf("Platform Number?\n");
	scanf("%d", &PLATFORM_NUM);
	printf("Device Number?\n");
	scanf("%d", &DEVICE_NUM);*/
	
	// Parse the command line arguments
	if (argc > 1) {
		PLATFORM_NUM = atoi(argv[1]);
		DEVICE_NUM = atoi(argv[2]);
		FILE_COUNT = atoi(argv[3]);
		PRINT_IPOINTS = atoi(argv[4]);
	} else {
		PLATFORM_NUM = 0;
		DEVICE_NUM = 0;
		FILE_COUNT = 1;
		PRINT_IPOINTS = false;
	}

	// Get all platforms and select platform
	clGetPlatformIDs(5, NULL, &platformCount);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	error = clGetPlatformIDs(platformCount, platforms, NULL);
	checkError(error, "Query for platforms failed");

	platformID = platforms[PLATFORM_NUM];
	printPlatformInfo(platformID);

	// Get devices for platform and select device
	error = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_ALL, 2, devices, NULL);
	checkError(error, "Query for devices failed");

	deviceID = devices[DEVICE_NUM];
	printDeviceInfo(deviceID);

	cl_ulong* value;
	size_t valueSize;
	clGetDeviceInfo(deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 0, NULL, &valueSize);
	value = (cl_ulong*)malloc(valueSize);
	clGetDeviceInfo(deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, valueSize, value, NULL);
	printf("\nMax mem: %lu", value);

	const cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties> (platformID),
		0, 0
	};

	context = clCreateContext(contextProperties, 1, &deviceID, NULL, NULL, &error);

	//commandQueue = clCreateCommandQueueWithProperties(context, deviceID, 0, &error);
	commandQueue = clCreateCommandQueue(context, deviceID, 0, &error);
	checkError(error, "Command queue creation failed");

	// Initialize all programs and kernels

	createProgram("./Hessian.cl", hessianProgram);
	createKernel("buildResponseLayer", hessianProgram, responseKernel);
	createKernel("getIPoints", hessianProgram, ipointsKernel);

	//size_t size;
	//clGetProgramInfo(hessianProgram, deviceID, CL_PROGRAM_BINARY_SIZES, 0, &size, NULL);


	//createProgram("./Desc.cl", descProgram);
	//createKernel("computeDesc", descProgram, descKernel);

	std::ofstream results;
	results.open("results.csv", std::ofstream::out | std::ofstream::trunc);

	for (int fileIndex = 0; fileIndex < FILE_COUNT; ++fileIndex) {

		// Read in the raw image
		std::stringstream sstm;
		sstm << "./img/test" << fileIndex << ".bmp";
		float *export2d = loadImage(sstm.str().c_str());

		std::cout << "\n================================ File: test" << fileIndex << ".bmp ================================" << "\n";
		std::cout << "Size: " << rawImageWidth << " x " << rawImageHeight << "\n";

		results << "File: " << sstm.str() << "\n";
		results << "Size: " << rawImageWidth << " x " << rawImageHeight << "\n";

		for (int run = 0; run < 10; ++run) {

			init();

			clock_t start = clock();

			// Compute Integral Image
			/*d_inputImage = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, rawImageWidth * rawImageHeight * sizeof(float), export2d, &error);
			checkError(error, "");
			d_integralImage = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, rawImageWidth * rawImageHeight * sizeof(float), NULL, &error);
			checkError(error, "");

			error = clSetKernelArg(integralKernel, 0, sizeof(cl_uint), &rawImageWidth);
			checkError(error, "");
			error = clSetKernelArg(integralKernel, 1, sizeof(cl_uint), &rawImageHeight);
			checkError(error, "");
			error = clSetKernelArg(integralKernel, 2, sizeof(cl_mem), &d_inputImage);
			checkError(error, "");
			error = clSetKernelArg(integralKernel, 3, sizeof(cl_mem), &d_integralImage);
			checkError(error, "");


			workerCount = 1;
			error = clEnqueueNDRangeKernel(commandQueue,
				integralKernel,
				1,
				NULL,
				&workerCount,
				&workerCount,
				0, NULL, NULL);
			checkError(error, "");

			error = clFinish(commandQueue);
			checkError(error, "");

			*/

			createIntegralImage(rawImageWidth, rawImageHeight, export2d, h_integralImage);

			d_integralImage = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, rawImageWidth * rawImageHeight * sizeof(float), h_integralImage, &error);
			checkError(error, "Creating integral image buffer failed");

			//std::cout << "\nIntegral finished" << "\n";
			clock_t endIntegral = clock();

			// Copy the integral image to host intImage
			/*float *intImage = new float[rawImageHeight * rawImageWidth];
			clEnqueueReadBuffer(commandQueue, d_integralImage, true, 0, rawImageWidth * rawImageHeight * sizeof(float), intImage, 0, NULL, NULL);

			for (int rowcount = 0; rowcount < rawImageHeight; rowcount++) {
				for (int colcount = 0; colcount < rawImageWidth; colcount++) {
					printf(" %f", intImage[rawImageWidth*rowcount + colcount]);
				}
				printf("\n");
			}*/

			//error = clFinish(commandQueue);
			//checkError(error, "");


			/*Calculate hessian responses*/

			clock_t startResponse = clock();

			// Initialize the ResponseLayers in the response map
			initResponseMap();

			// Sets the various filter properties based on the response map
			buildFilterProperties(h_responseMap, h_filterWidth, h_filterBorder, h_filterLobe, h_filterStep, h_normaliseFactor);

			// Create the arguments for response kernel
			d_responseMap = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint4) * 12, h_responseMap, &error);
			checkError(error, "");
			d_filterWidth = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 12, h_filterWidth, &error);
			checkError(error, "");
			d_filterBorder = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 12, h_filterBorder, &error);
			checkError(error, "");
			d_filterLobe = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 12, h_filterLobe, &error);
			checkError(error, "");
			d_filterStep = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 12, h_filterStep, &error);
			checkError(error, "");
			d_normaliseFactor = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * 12, h_normaliseFactor, &error);
			checkError(error, "");

			//Set Response Kernel Arg
			error = clSetKernelArg(responseKernel, 0, sizeof(cl_mem), &d_integralImage);
			checkError(error, "");
			error = clSetKernelArg(responseKernel, 1, sizeof(cl_mem), &d_responseMap);
			checkError(error, "");
			error = clSetKernelArg(responseKernel, 2, sizeof(cl_mem), &d_filterWidth);
			checkError(error, "");
			error = clSetKernelArg(responseKernel, 3, sizeof(cl_mem), &d_filterBorder);
			checkError(error, "");
			error = clSetKernelArg(responseKernel, 4, sizeof(cl_mem), &d_filterLobe);
			checkError(error, "");
			error = clSetKernelArg(responseKernel, 5, sizeof(cl_mem), &d_filterStep);
			checkError(error, "");
			error = clSetKernelArg(responseKernel, 6, sizeof(cl_mem), &d_normaliseFactor);
			checkError(error, "");
			error = clSetKernelArg(responseKernel, 7, sizeof(cl_uint), &rawImageWidth);
			checkError(error, "");
			error = clSetKernelArg(responseKernel, 8, sizeof(cl_uint), &rawImageHeight);
			checkError(error, "");
			error = clSetKernelArg(responseKernel, 9, sizeof(cl_mem), &d_responseBuffer);
			checkError(error, "");
			error = clSetKernelArg(responseKernel, 10, sizeof(cl_mem), &d_laplacianBuffer);
			checkError(error, "");

			// Run the response kernel
			//workerCount = rawImageHeight;
			size_t responseKernelWorkSize;
			clGetKernelWorkGroupInfo(responseKernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &responseKernelWorkSize, NULL);
			checkError(error, "Get response kernel work group size failed");

			responseKernelWorkSize = (responseKernelWorkSize < rawImageHeight) ? responseKernelWorkSize : rawImageHeight;

			error = clEnqueueNDRangeKernel(commandQueue,
				responseKernel,
				1,
				NULL,
				&responseKernelWorkSize,
				&responseKernelWorkSize,
				0, NULL, NULL);
			checkError(error, "Enqueue response kernel failed");

			//error = clFinish(commandQueue);
			//checkError(error, "Waiting for response kernel to finish failed");

			//clEnqueueReadBuffer(commandQueue, d_laplacianBuffer, CL_TRUE, 0, sizeof(cl_uint) * rawImageWidth * rawImageHeight * 12, h_laplacianBuffer, 0, NULL, NULL);
			//for (int i = 0; i < rawImageWidth * rawImageHeight * 12; ++i) {
			//	//if(h_responseBuffer[i] != 0.f)
			//		printf("%u, ", h_laplacianBuffer[i]);
			//}

			//std::cout << "\nResponse finished" << "\n";

			clock_t endResponse = clock();


			/*Get the IPoints*/

			clock_t startIPoint = clock();

			d_iptsIndex = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &h_iptsCount, &error);
			checkError(error, "");

			// Set the kernel arguments that are shared across intervals (workgroups)
			error = clSetKernelArg(ipointsKernel, 0, sizeof(cl_uint), &rawImageWidth);
			checkError(error, "");
			error = clSetKernelArg(ipointsKernel, 1, sizeof(cl_uint), &rawImageHeight);
			checkError(error, "");
			error = clSetKernelArg(ipointsKernel, 2, sizeof(cl_mem), &d_responseMap);
			checkError(error, "");
			error = clSetKernelArg(ipointsKernel, 3, sizeof(cl_mem), &d_ipts);
			checkError(error, "");
			error = clSetKernelArg(ipointsKernel, 4, sizeof(cl_mem), &d_iptsIndex);
			checkError(error, "");
			error = clSetKernelArg(ipointsKernel, 5, sizeof(cl_mem), &d_responseBuffer);
			checkError(error, "");
			error = clSetKernelArg(ipointsKernel, 6, sizeof(cl_mem), &d_laplacianBuffer);
			checkError(error, "");

			// Iterate over all intervals and run a kernel
			clGetKernelWorkGroupInfo(ipointsKernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &ipointsKernelLocalSize, NULL);
			checkError(error, "");
			ipointsKernelLocalSize = (ipointsKernelLocalSize < rawImageHeight) ? ipointsKernelLocalSize : rawImageHeight;
			ipointsKernelGlobalSize = ipointsKernelLocalSize * 10;

			error = clEnqueueNDRangeKernel(commandQueue,
				ipointsKernel,
				1,
				NULL,
				&ipointsKernelGlobalSize,
				&ipointsKernelLocalSize,
				0, NULL, NULL);
			checkError(error, "Enqueue ipoints kernel failed");

			cl_event events[1];
			error = clEnqueueReadBuffer(commandQueue, d_iptsIndex, CL_TRUE, 0, sizeof(cl_uint), &h_iptsCount, 0, NULL, NULL);
			checkError(error, "");
			h_iptsTempCopy = new IPoint_small[h_iptsCount];
			error = clEnqueueReadBuffer(commandQueue, d_ipts, CL_FALSE, 0, sizeof(IPoint_small) * h_iptsCount, h_iptsTempCopy, 0, NULL, events);
			checkError(error, "");

			//error = clEnqueueReadBuffer(commandQueue, d_ipts, CL_TRUE, 0, sizeof(IPoint) * rawImageHeight * rawImageWidth, h_ipts, 0, NULL, NULL);
			//checkError(error, "");
			//for (unsigned int i = 0; i < rawImageHeight * rawImageWidth; ++i) {
			//	printf("\n(%f, %f)\n", h_ipts[i].x, h_ipts[i].y);
			//	printf("Scale: %f\n", h_ipts[i].scale);
			//	printf("Laplacian: %u\n", h_ipts[i].laplacian);
			//}

			//std::cout << "\nIPoints finished" << "\n";

			clock_t endIPoint = clock();


			/* Get the descriptors */

			//error = clWaitForEvents(1, events);
			//checkError(error, "");
			//printf("copy");

			clock_t startDesc = clock();

			h_iptsDesc = new IPoint[h_iptsCount]();

			clWaitForEvents(1, events);
			//delete[] h_ipts;
			for (unsigned int i = 0; i < h_iptsCount; ++i) {
				h_iptsDesc[i].x = h_iptsTempCopy[i].x;
				h_iptsDesc[i].y = h_iptsTempCopy[i].y;
				h_iptsDesc[i].scale = h_iptsTempCopy[i].scale;
				h_iptsDesc[i].laplacian = h_iptsTempCopy[i].laplacian;

				getOrientation(&h_iptsDesc[i]);
				getDescriptor(&h_iptsDesc[i]);
			}

			//std::cout << "\nDescriptors finished" << "\n";

			//error = clSetKernelArg(descKernel, 0, sizeof(cl_mem), &d_integralImage);
			//checkError(error, "");
			//error = clSetKernelArg(descKernel, 1, sizeof(cl_mem), &d_ipts);
			//checkError(error, "");
			//error = clSetKernelArg(descKernel, 2, sizeof(cl_mem), &d_iptsIndex);
			//checkError(error, "");
			//error = clSetKernelArg(descKernel, 3, sizeof(cl_uint), &rawImageWidth);
			//checkError(error, "");
			//error = clSetKernelArg(descKernel, 4, sizeof(cl_uint), &rawImageHeight);
			//checkError(error, "");
			//
			////workerCount = iptsCount;
			//size_t descKernelWorkSize;
			//clGetKernelWorkGroupInfo(descKernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &descKernelWorkSize, NULL);
			//checkError(error, "");

			//clWaitForEvents(1, events);
			//descKernelWorkSize = (descKernelWorkSize < h_iptsCount) ? descKernelWorkSize : h_iptsCount;

			//error = clEnqueueNDRangeKernel(commandQueue,
			//	descKernel,
			//	1,
			//	NULL,
			//	&descKernelWorkSize,
			//	&descKernelWorkSize,
			//	0, NULL, NULL);
			//checkError(error, "");

			//error = clFinish(commandQueue);
			//checkError(error, "");

			clock_t end = clock();
			clock_t endDesc = clock();


			//IPoint *h_iptsDesc = new IPoint[h_iptsCount];
			//error = clEnqueueReadBuffer(commandQueue, d_ipts, CL_TRUE, 0, sizeof(IPoint) * h_iptsCount, h_iptsDesc, 0, NULL, NULL);
			//checkError(error, "");

			if (PRINT_IPOINTS) {
				for (unsigned int i = 0; i < h_iptsCount; ++i) {
					printf("\nIPoint %d: \n", i + 1);
					printf("(%f, %f) \n", round(h_iptsDesc[i].x), round(h_iptsDesc[i].y));
					printf("Orientation: %f \n", h_iptsDesc[i].orientation);
					printf("Laplacian: %d \n", h_iptsDesc[i].laplacian);
					printf("Descriptors: ");
					for (int j = 0; j < 64; ++j) {
						printf("%f ", h_iptsDesc[i].descriptors[j]);
					}
					printf("\n");
				}
			}

			std::cout << "\nRun " << run << "\n";
			printf("Response Worksize: %d\n", responseKernelWorkSize);
			printf("IPoints Global Size: %d, Local Size: %d\n", ipointsKernelGlobalSize, ipointsKernelLocalSize);

			printf("Integral took: %f seconds\n", float(endIntegral - start) / CLOCKS_PER_SEC);
			printf("Response took: %f seconds\n", float(endResponse - startResponse) / CLOCKS_PER_SEC);
			printf("IPoints took: %f seconds\n", float(endIPoint - startIPoint) / CLOCKS_PER_SEC);
			printf("Descriptors took: %f seconds\n", float(endDesc - startDesc) / CLOCKS_PER_SEC);

			printf("OpenCL Surf Found: %d interest points\n", h_iptsCount);
			printf("OpenCL Surf took: %f seconds\n", float(end - start) / CLOCKS_PER_SEC);

			results << float(end - start) / CLOCKS_PER_SEC << ",";

			cleanup();

		}

		results << "\n\n";
	}

	return 0;
}


inline void init() {
	// Integral
	h_integralImage = new float[rawImageHeight * rawImageWidth];

	// Response
	// Used to store all the responses after calculation
	//h_responseBuffer = new cl_float[sizeof(cl_float) * rawImageWidth * rawImageHeight * 12]();
	//h_laplacianBuffer = new cl_uint[sizeof(cl_float) * rawImageWidth * rawImageHeight * 12]();
	d_responseBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * rawImageWidth * rawImageHeight * 12, NULL, &error);
	checkError(error, "");
	d_laplacianBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * rawImageWidth * rawImageHeight * 12, NULL, &error);
	checkError(error, "");

	// IPoints
	h_iptsCount = 0;
	h_ipts = new IPoint_small[rawImageHeight * rawImageWidth]();
	d_ipts = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(IPoint_small) * rawImageHeight * rawImageWidth, h_ipts, &error);
	checkError(error, "");
}


inline void cleanup() {
	delete[] h_integralImage;
	delete[] h_responseMap;
	//delete[] h_responseBuffer;
	//delete[] h_laplacianBuffer;
	delete[] h_ipts;
	delete[] h_iptsDesc;
	delete[] h_iptsTempCopy;

	error = clReleaseMemObject(d_ipts);
	checkError(error, "");
	error = clReleaseMemObject(d_iptsIndex);
	checkError(error, "");
	error = clReleaseMemObject(d_laplacianBuffer);
	checkError(error, "");
	error = clReleaseMemObject(d_responseBuffer);
	checkError(error, "");
	error = clReleaseMemObject(d_responseMap);
	checkError(error, "");
	error = clReleaseMemObject(d_integralImage);
	checkError(error, "");
	error = clReleaseMemObject(d_filterBorder);
	checkError(error, "");
	error = clReleaseMemObject(d_filterLobe);
	checkError(error, "");
	error = clReleaseMemObject(d_filterStep);
	checkError(error, "");
	error = clReleaseMemObject(d_filterWidth);
	checkError(error, "");
	error = clReleaseMemObject(d_normaliseFactor);
	checkError(error, "");
	//error = clReleaseMemObject(d_inputImage);
	//checkError(error, "");
}


inline float BoxIntegralImage(int row, int col, int rows, int cols) {
	//row, col is the upper left corner of the box
	//row+rows, col+cols is the lower right corner of the box

	// Subtract by 1 because it is inclusive
	int r1 = std::min(row, (int)rawImageHeight) - 1;
	int c1 = std::min(col, (int)rawImageWidth) - 1;
	int r2 = std::min(row + rows, (int)rawImageHeight) - 1;
	int c2 = std::min(col + cols, (int)rawImageWidth) - 1;

	float A = 0, B = 0, C = 0, D = 0;

	//Top left
	if (r1 >= 0 && c1 >= 0) {
		A = h_integralImage[r1*rawImageWidth + c1];
	}

	//Top right
	if (r1 >= 0 && c2 >= 0) {
		B = h_integralImage[r1*rawImageWidth + c2];
	}

	//Bottom left
	if (r2 >= 0 && c1 >= 0) {
		C = h_integralImage[r2*rawImageWidth + c1];
	}

	//Bottom right
	if (r2 >= 0 && c2 >= 0) {
		D = h_integralImage[r2*rawImageWidth + c2];
	}

	return std::max(0.f, A - B - C + D);
}

inline float getAngle(float x, float y) {
	if (x > 0.f && y >= 0.f)
		return atan2(y , x);

	if (x < 0.f && y >= 0.f)
		return M_PI - atan2(-y, x);

	if (x < 0.f && y < 0.f)
		return M_PI + atan2(y, x);

	if (x > 0.f && y < 0.f)
		return 2 * M_PI - atan2(-y, x);

	return 0;
}

//! Calculate the value of the 2d gaussian at x,y
inline float gaussian(int x, int y, float sig)
{
	return (1.0f / (2.0f*M_PI*sig*sig)) * exp(-(x*x + y*y) / (2.0f*sig*sig));
}

//! Calculate the value of the 2d gaussian at x,y
inline float gaussian(float x, float y, float sig)
{
	return 1.0f / (2.0f*M_PI*sig*sig) * exp(-(x*x + y*y) / (2.0f*sig*sig));
}

inline float haarX(int row, int column, int s) {
	return BoxIntegralImage(row - s / 2, column, s, s / 2) - BoxIntegralImage(row - s / 2, column - s / 2, s, s / 2);

}

inline float haarY(int row, int column, int s) {
	return BoxIntegralImage(row, column - s / 2, s / 2, s) - BoxIntegralImage(row - s / 2, column - s / 2, s / 2, s);
}

//! Round float to nearest integer
inline int fRound(float flt)
{
  return (int) floor(flt+0.5f);
}

//! Assign the supplied Ipoint an orientation
void getOrientation(IPoint *ipt)
{
	//IPoint *ipt = &h_ipts[descIptIndex];
	float gauss = 0.f, scale = ipt->scale;
	const int s = fRound(scale), r = fRound(ipt->y), c = fRound(ipt->x);
	std::vector<float> resX(109), resY(109), Ang(109);
	const int id[] = { 6,5,4,3,2,1,0,1,2,3,4,5,6 };

	int idx = 0;
	// calculate haar responses for points within radius of 6*scale
	for (int i = -6; i <= 6; ++i)
	{
		for (int j = -6; j <= 6; ++j)
		{
			if (i*i + j*j < 36)
			{
				gauss = static_cast<float>(gauss25[id[i + 6]][id[j + 6]]);  // could use abs() rather than id lookup, but this way is faster
				resX[idx] = gauss * haarX(r + j*s, c + i*s, 4 * s);
				resY[idx] = gauss * haarY(r + j*s, c + i*s, 4 * s);
				Ang[idx] = getAngle(resX[idx], resY[idx]);
				//printf("\n%d, %f", index, resX[idx]);
				++idx;
			}
		}
	}

	// calculate the dominant direction 
	float sumX = 0.f, sumY = 0.f;
	float max = 0.f, orientation = 0.f;
	float ang1 = 0.f, ang2 = 0.f;

	// loop slides pi/3 window around feature point
	for (ang1 = 0.f; ang1 < 2 * M_PI; ang1 += 0.15f) {
		ang2 = (ang1 + M_PI / 3.0f > 2 * M_PI ? ang1 - 5.0f*M_PI / 3.0f : ang1 + M_PI / 3.0f);
		sumX = sumY = 0.f;
		for (unsigned int k = 0; k < Ang.size(); ++k)
		{
			// get angle from the x-axis of the sample point
			const float & ang = Ang[k];

			// determine whether the point is within the window
			if (ang1 < ang2 && ang1 < ang && ang < ang2)
			{
				sumX += resX[k];
				sumY += resY[k];
			}
			else if (ang2 < ang1 &&
				((ang > 0.f && ang < ang2) || (ang > ang1 && ang < 2.f * M_PI)))
			{
				sumX += resX[k];
				sumY += resY[k];
			}
		}

		// if the vector produced from this window is longer than all 
		// previous vectors then this forms the new dominant direction
		if (sumX*sumX + sumY*sumY > max)
		{
			// store largest orientation
			max = sumX*sumX + sumY*sumY;
			orientation = getAngle(sumX, sumY);
			//printf("%f \n", orientation);
		}
	}

	// assign orientation of the dominant response vector
	ipt->orientation = orientation;
}


//! Get the modified descriptor. See Agrawal ECCV 08
//! Modified descriptor contributed by Pablo Fernandez
void getDescriptor(IPoint *ipt) {
	//IPoint *ipt = &h_ipts[descIptIndex];

	int y, x, sample_x, sample_y, count = 0;
	int i = 0, ix = 0, j = 0, jx = 0, xs = 0, ys = 0;
	float scale, *desc, dx, dy, mdx, mdy, co, si;
	float gauss_s1 = 0.f, gauss_s2 = 0.f;
	float rx = 0.f, ry = 0.f, rrx = 0.f, rry = 0.f, len = 0.f;
	float cx = -0.5f, cy = 0.f; //Subregion centers for the 4x4 gaussian weighting

	scale = ipt->scale;
	x = fRound(ipt->x);
	y = fRound(ipt->y);
	desc = ipt->descriptors;

	co = cos(ipt->orientation);
	si = sin(ipt->orientation);

	i = -8;

	//Calculate descriptor for this interest point
	while (i < 12)
	{
		j = -8;
		i = i - 4;

		cx += 1.f;
		cy = -0.5f;

		while (j < 12)
		{
			dx = dy = mdx = mdy = 0.f;
			cy += 1.f;

			j = j - 4;

			ix = i + 5;
			jx = j + 5;

			xs = fRound(x + (-jx*scale*si + ix*scale*co));
			ys = fRound(y + (jx*scale*co + ix*scale*si));

			for (int k = i; k < i + 9; ++k)
			{
				for (int l = j; l < j + 9; ++l)
				{
					//Get coords of sample point on the rotated axis
					sample_x = fRound(x + (-l*scale*si + k*scale*co));
					sample_y = fRound(y + (l*scale*co + k*scale*si));

					//Get the gaussian weighted x and y responses
					gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f*scale);
					rx = haarX(sample_y, sample_x, 2 * fRound(scale));
					ry = haarY(sample_y, sample_x, 2 * fRound(scale));

					//Get the gaussian weighted x and y responses on rotated axis
					rrx = gauss_s1*(-rx*si + ry*co);
					rry = gauss_s1*(rx*co + ry*si);

					dx += rrx;
					dy += rry;
					mdx += fabs(rrx);
					mdy += fabs(rry);

				}
			}

			//Add the values to the descriptor vector
			gauss_s2 = gaussian(cx - 2.0f, cy - 2.0f, 1.5f);

			desc[count++] = dx*gauss_s2;
			desc[count++] = dy*gauss_s2;
			desc[count++] = mdx*gauss_s2;
			desc[count++] = mdy*gauss_s2;

			len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy) * gauss_s2*gauss_s2;

			j += 9;
		}
		i += 9;
	}

	//Convert to Unit Vector
	len = sqrt(len);
	for (int i = 0; i < 64; ++i)
		desc[i] /= len;

}


inline void createIntegralImage(unsigned int width, unsigned int height, float *imageData, float *result) {
	float rs = 0.0f;

	for (unsigned int x = 0; x < width; x++) {
		rs += imageData[x];
		result[x] = rs;
	}

	//barrier(CLK_LOCAL_MEM_FENCE); // to be safe i think

	// remaining cells are sum above and to the left
	for (unsigned int y = 1; y < height; y++) { // iterate over each row
		rs = 0.0f; // stores accumulative sum of row data
		for (unsigned int x = 0; x < width; x++) { // iterate over each item in the row
			rs += imageData[y*width + x]; // accumulate row sum
			result[y*width + x] = rs + result[(y - 1)*width + x]; // add the integral value of the pixel above
		}
	}
}


inline void printPlatformInfo(cl_platform_id platform) {
	// Print out platform info
	char* info;
	size_t infoSize;
	const char* attributeNames[5] = { "Name", "Vendor", "Version", "Profile", "Extensions" };
	const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
		CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
	const int attributeCount = sizeof(attributeNames) / sizeof(char*);
	printf("\n %d. Platform \n", 1);

	for (int j = 0; j < attributeCount; ++j) {

		// get platform attribute value size
		clGetPlatformInfo(platform, attributeTypes[j], 0, NULL, &infoSize);
		info = (char*)malloc(infoSize);

		// get platform attribute value
		clGetPlatformInfo(platform, attributeTypes[j], infoSize, info, NULL);

		printf("  %d.%d %-11s: %s\n", 1, j + 1, attributeNames[j], info);
		free(info);

	}
	printf("\n");
}


inline void printDeviceInfo(cl_device_id device) {
	char* value;
	size_t valueSize;
	cl_uint maxComputeUnits;

	// print device name
	clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
	printf("Device: %s\n", value);
	free(value);

	// print hardware device version
	clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(device, CL_DEVICE_VERSION, valueSize, value, NULL);
	printf(" %d Hardware version: %s\n", 1, value);
	free(value);

	// print software driver version
	clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(device, CL_DRIVER_VERSION, valueSize, value, NULL);
	printf(" %d Software version: %s\n", 2, value);
	free(value);

	// print c version supported by compiler for device
	clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
	printf(" %d OpenCL C version: %s\n", 3, value);
	free(value);

	// print parallel compute units
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
	printf(" %d Parallel compute units: %d\n", 4, maxComputeUnits);

	printf("\n");
}


// Used to create an opencl program
inline void createProgram(char *filepath, cl_program &program) {
	cl_int ErrorCode;

	//load the cl file
	std::fstream Instream(filepath, std::fstream::in | std::fstream::binary);
	assert(Instream.is_open());

	//get the length
	Instream.seekg(0, std::fstream::end);
	size_t  FileSize = (size_t)Instream.tellg();
	Instream.seekg(0, std::fstream::beg);

	//read the sourcecode
	char* SourceCode = new char[FileSize + 1];
	Instream.read(SourceCode, FileSize);
	SourceCode[FileSize] = '\0';
	Instream.close();

	const char* src = const_cast<const char*>(SourceCode);

	std::cout << "\nBuilding program " << filepath << "\n";
	program = clCreateProgramWithSource(context, 1, &src, &FileSize, &ErrorCode);
	assert(ErrorCode == CL_SUCCESS);

	ErrorCode = clBuildProgram(program, 1, &deviceID, "-cl-std=CL1.2", NULL, NULL);
	
	//if (ErrorCode != CL_SUCCESS) {
		std::cout << "Build log: \n" << getErrorString(ErrorCode) << "\n";
		// Shows the log
		char* build_log;
		size_t log_size;
		// First call to know the proper size
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, NULL, NULL, &log_size);
		build_log = new char[1000000];
		// Second call to get the log
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 1000000, build_log, &log_size);
		std::cout << build_log << "\n";
		delete[] build_log;
	//}
	assert(ErrorCode == CL_SUCCESS);

	delete[] SourceCode;
}

inline void createKernel(char *kernelName, cl_program &program, cl_kernel &kernel) {
	cl_int ErrorCode;
	kernel = clCreateKernel(program, kernelName, &ErrorCode);
	checkError(error, "Create kernel failed");
}

inline float* loadImage(const char* fileName) {
	std::ios_base::sync_with_stdio(false);

	std::basic_ifstream<unsigned char> inputFile;
	inputFile.open(fileName, std::ifstream::in | std::ifstream::binary);

	unsigned char imageinfo[54];
	inputFile.read(imageinfo, 54);
	rawImageWidth = *(int*)&imageinfo[18];
	rawImageHeight = abs(*(int*)&imageinfo[22]);
	int row_padded = (rawImageWidth * 3 + 3) & (~3);
	unsigned char* data = new unsigned char[3 * rawImageWidth*rawImageHeight];
	//printf("imageheight: %d imagewidth: %d\n", imageHeight, imageWidth);
	//printf("\n debug %d %d\n", row_padded, imageWidth);
	unsigned char* rubbish = new unsigned char[row_padded - rawImageWidth];
	for (unsigned int i = 0; i < rawImageHeight; ++i) {
		inputFile.read(&data[i * 3 * rawImageWidth], rawImageWidth * 3);
		inputFile.read(rubbish, row_padded - 3 * rawImageWidth);

	}
	/*The color of pixel(i, j) is stored at data[j * width + i], data[j * width + i + 1] and data[j * width + i + 2].*/



	//for (int i = 0; i < 100; ++i) {
	//	printf("%d %hhx %hhx %hhx\n", i,data[i*3], data[i*3+1], data[i*3+2]);
	//}



	float *export2d = new float[rawImageWidth*rawImageHeight];

	for (unsigned int rowcount = 0; rowcount < rawImageHeight; rowcount++) {
		for (unsigned int colcount = 0; colcount < rawImageWidth; colcount++) {
			//printf("%x %x %x\n", data[3 * rawImageWidth*rowcount + 3 * colcount], data[3 * rawImageWidth*rowcount + 3 * colcount + 1], data[3 * rawImageWidth*rowcount + 3 * colcount + 2]);
			//export2d[rawImageWidth*rowcount + colcount] = (data[3 * rawImageWidth*rowcount + 3 * colcount] * 0.114f + data[3 * rawImageWidth*rowcount + 3 * colcount + 1] * 0.587f + data[3 * rawImageWidth*rowcount + 3 * colcount + 2] * 0.299f)/255.0f;
			export2d[rawImageWidth*rowcount + colcount] = (data[3 * rawImageWidth*rowcount + 3 * colcount] * 0.114f + data[3 * rawImageWidth*rowcount + 3 * colcount + 1] * 0.587f + data[3 * rawImageWidth*rowcount + 3 * colcount + 2] * 0.299f)/255.0f;
			//printf(" %f", export2d[rawImageWidth*rowcount + colcount]);
		}
		//printf("\n");
	}

	delete[] data;
	delete[] rubbish;

	return export2d;

}

	/*void loadImage() {
		FILE *inputfile = fopen("test.bmp", "rb");
		unsigned char imageinfo[54];

		fread(imageinfo, sizeof(unsigned char), 54, inputfile);
		int imageWidth = *(int*)&imageinfo[18];
		int imageHeight = *(int*)&imageinfo[22];
		int row_padded = (imageWidth * 3 + 3) & (~3);
		unsigned char* data = new unsigned char[3 * imageWidth*imageHeight];
		unsigned char tmp;
		//printf("imageheight: %d imagewidth: %d\n", imageHeight, imageWidth);
		//printf("\n debug %d %d\n", row_padded, imageWidth);
		unsigned char* rubbish = new unsigned char[row_padded - imageWidth];
		for (int i = 0; i < imageHeight; ++i) {
			fread(&data[i * 3 * imageWidth], 1, imageWidth * 3, inputfile);
			fread(rubbish, 1, row_padded - 3 * imageWidth, inputfile);


		}
		/*The color of pixel(i, j) is stored at data[j * width + i], data[j * width + i + 1] and data[j * width + i + 2].



		//for (int i = 0; i < 100; ++i) {
		//	printf("%d %hhx %hhx %hhx\n", i,data[i*3], data[i*3+1], data[i*3+2]);
		//}



		h_rawImage = new float[imageWidth*imageHeight];

		for (int i = 0; i<imageWidth; ++i) {

			for (int j = 0; j<imageHeight; ++j) {
				h_rawImage[i*imageWidth + j] = data[j*imageWidth + i] * 0.30 + data[j*imageWidth + 1 + i] * 0.59 + data[j*imageWidth + i + 2] * 0.11;
			}
		}

		rawImageWidth = imageWidth;
		rawImageHeight = imageHeight;

		//cl_image_desc desc;
		//memset(&desc, '\0', sizeof(cl_image_desc));
		//desc.image_type = CL_MEM_OBJECT_IMAGE2D;
		//desc.image_width = imageWidth;
		//desc.image_height = imageHeight;
		//desc.mem_object = NULL; // or someBuf;

		//cl_int error;
		//cl_image_format format;
		//format.image_channel_order = CL_INTENSITY;
		//format.image_channel_data_type = CL_FLOAT;

		//return clCreateImage(context, 0, &format, &desc, export2d, &error);
	}*/

	/*
	float* integral(float *img, int height, int width) {
		int size = height * width;
		float *integralImg = new float[size]; // create array for the integral image

											  // first row only
		float rs = 0.0f;
		for (int j = 0; j < width; ++j) {
			rs += img[j];
			integralImg[j] = rs;
		}

		// remaining cells are sum above and to the left
		for (int i = 1; i < height; ++i) { // iterate over each row
			rs = 0.0f; // stores accmmulative sum of row data
			for (int j = 0; j < width; ++j) { // iterate over each item in the row
				rs += img[i*width + j]; // add value of item to rs
				integralImg[i*width + j] = rs + integralImg[(i - 1)*width + j]; // add the value of the cell above and rs
			}
		}

		return integralImg;
	}



	//! Computes the sum of pixels within the rectangle specified by the top-left start
	//! co-ordinate and size
	float BoxIntegral(float *img, int row, int col, int rows, int cols, int height, int width)
	{
		// The subtraction by one for row/col is because row/col is inclusive.
		int r1 = std::min(row, height) - 1;
		int c1 = std::min(col, width) - 1;
		int r2 = std::min(row + rows, height) - 1;
		int c2 = std::min(col + cols, width) - 1;

		float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
		if (r1 >= 0 && c1 >= 0) A = img[r1*width + c1];
		if (r1 >= 0 && c2 >= 0) B = img[r1*width + c2];
		if (r2 >= 0 && c1 >= 0) C = img[r2*width + c1];
		if (r2 >= 0 && c2 >= 0) D = img[r2*width + c2];

		return std::max(0.f, A - B - C + D);
	}
	*/


inline void initResponseMap() {
	//cl_int4 width height step filter
	//x y z w	//h_responseMap = new ResponseLayer[12];

	h_responseMap = new cl_int4[12];
	int w = (rawImageWidth / INITSAMPLE);
	int h = (rawImageHeight / INITSAMPLE);
	int s = 2;
	if (OCTAVES >= 1)
	{
		h_responseMap[0].x = w; h_responseMap[0].y = h; h_responseMap[0].z = s; h_responseMap[0].w = 9;
		h_responseMap[1].x = w; h_responseMap[1].y = h; h_responseMap[1].z = s; h_responseMap[1].w = 15;
		h_responseMap[2].x = w; h_responseMap[2].y = h; h_responseMap[2].z = s; h_responseMap[2].w = 21;
		h_responseMap[3].x = w; h_responseMap[3].y = h; h_responseMap[3].z = s; h_responseMap[3].w = 27;
	}

	if (OCTAVES >= 2)
	{
		h_responseMap[4].x = w / 2; h_responseMap[4].y = h / 2; h_responseMap[4].z = s * 2; h_responseMap[4].w = 39;
		h_responseMap[5].x = w / 2; h_responseMap[5].y = h / 2; h_responseMap[5].z = s * 2; h_responseMap[5].w = 51;
	}

	if (OCTAVES >= 3)
	{
		h_responseMap[6].x = w / 4; h_responseMap[6].y = h / 4; h_responseMap[6].z = s * 4; h_responseMap[6].w = 75;
		h_responseMap[7].x = w / 4; h_responseMap[7].y = h / 4; h_responseMap[7].z = s * 4; h_responseMap[7].w = 99;
	}
	if (OCTAVES >= 4)
	{
		h_responseMap[8].x = w / 8; h_responseMap[8].y = h / 8; h_responseMap[8].z = s * 8; h_responseMap[8].w = 147;
		h_responseMap[9].x = w / 8; h_responseMap[9].y = h / 8; h_responseMap[9].z = s * 8; h_responseMap[9].w = 195;
	}

	if (OCTAVES >= 5)
	{
		h_responseMap[10].x = w / 16; h_responseMap[10].y = h / 16; h_responseMap[10].z = s * 16; h_responseMap[10].w = 291;
		h_responseMap[11].x = w / 16; h_responseMap[11].y = h / 16; h_responseMap[11].z = s * 16; h_responseMap[11].w = 387;
	}
}


// Set the various filter properties for all 12 layers
inline void buildFilterProperties(cl_int4 *responseMap, cl_int *filterWidth, cl_int *filterBorder, cl_int *filterLobe, cl_int *filterStep, cl_float *normaliseFactor) {

	//for (int i = 0; i < 12; ++i) {
	//	filterWidth[i] = responseMap[i].filter;
	//	filterBorder[i] = ((filterWidth[i] - 1) / 2);
	//	filterLobe[i] = filterWidth[i] / 3;
	//	filterStep[i] = responseMap[i].step;
	//	normaliseFactor[i] = 1.0f*filterWidth[i] * filterWidth[i];
	//}

	//unrolled loop for lower overhead
	filterWidth[0] = responseMap[0].w;
	filterBorder[0] = ((filterWidth[0] - 1) / 2);
	filterLobe[0] = filterWidth[0] / 3;
	filterStep[0] = responseMap[0].z;
	normaliseFactor[0] = 1.0f*filterWidth[0] * filterWidth[0];

	filterWidth[1] = responseMap[1].w;
	filterBorder[1] = ((filterWidth[1] - 1) / 2);
	filterLobe[1] = filterWidth[1] / 3;
	filterStep[1] = responseMap[1].z;
	normaliseFactor[1] = 1.0f*filterWidth[1] * filterWidth[1];

	filterWidth[2] = responseMap[2].w;
	filterBorder[2] = ((filterWidth[2] - 1) / 2);
	filterLobe[2] = filterWidth[2] / 3;
	filterStep[2] = responseMap[2].z;
	normaliseFactor[2] = 1.0f*filterWidth[2] * filterWidth[2];

	filterWidth[3] = responseMap[3].w;
	filterBorder[3] = ((filterWidth[3] - 1) / 2);
	filterLobe[3] = filterWidth[3] / 3;
	filterStep[3] = responseMap[3].z;
	normaliseFactor[3] = 1.0f*filterWidth[3] * filterWidth[3];

	filterWidth[4] = responseMap[4].w;
	filterBorder[4] = ((filterWidth[4] - 1) / 2);
	filterLobe[4] = filterWidth[4] / 3;
	filterStep[4] = responseMap[4].z;
	normaliseFactor[4] = 1.0f*filterWidth[4] * filterWidth[4];

	filterWidth[5] = responseMap[5].w;
	filterBorder[5] = ((filterWidth[5] - 1) / 2);
	filterLobe[5] = filterWidth[5] / 3;
	filterStep[5] = responseMap[5].z;
	normaliseFactor[5] = 1.0f*filterWidth[5] * filterWidth[5];

	filterWidth[6] = responseMap[6].w;
	filterBorder[6] = ((filterWidth[6] - 1) / 2);
	filterLobe[6] = filterWidth[6] / 3;
	filterStep[6] = responseMap[6].z;
	normaliseFactor[6] = 1.0f*filterWidth[6] * filterWidth[6];

	filterWidth[7] = responseMap[7].w;
	filterBorder[7] = ((filterWidth[7] - 1) / 2);
	filterLobe[7] = filterWidth[7] / 3;
	filterStep[7] = responseMap[7].z;
	normaliseFactor[7] = 1.0f*filterWidth[7] * filterWidth[7];

	filterWidth[8] = responseMap[8].w;
	filterBorder[8] = ((filterWidth[8] - 1) / 2);
	filterLobe[8] = filterWidth[8] / 3;
	filterStep[8] = responseMap[8].z;
	normaliseFactor[8] = 1.0f*filterWidth[8] * filterWidth[8];

	filterWidth[9] = responseMap[9].w;
	filterBorder[9] = ((filterWidth[9] - 1) / 2);
	filterLobe[9] = filterWidth[9] / 3;
	filterStep[9] = responseMap[9].z;
	normaliseFactor[9] = 1.0f*filterWidth[9] * filterWidth[9];

	filterWidth[10] = responseMap[10].w;
	filterBorder[10] = ((filterWidth[10] - 1) / 2);
	filterLobe[10] = filterWidth[10] / 3;
	filterStep[10] = responseMap[10].z;
	normaliseFactor[10] = 1.0f*filterWidth[10] * filterWidth[10];

	filterWidth[11] = responseMap[11].w;
	filterBorder[11] = ((filterWidth[11] - 1) / 2);
	filterLobe[11] = filterWidth[11] / 3;
	filterStep[11] = responseMap[11].z;
	normaliseFactor[11] = 1.0f*filterWidth[11] * filterWidth[11];
}

// Print line, file name, and error code if there is an error. Exits the
// application upon error.
inline void _checkError(int line, const char *file, cl_int error, const char *msg) {
	// If not successful
	if (error != CL_SUCCESS) {
		// Print line and file
		printf("ERROR: ");
		printf(getErrorString(error));
		printf("\nLocation: %s:%d\n", file, line);

		// Print custom message.
		printf(msg);

		// Cleanup and bail.
		//cleanup();
		exit(error);
	}
}


inline const char* getErrorString(cl_int m_error)
{
	switch (m_error) {
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}