__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

typedef struct {
	float x;
	float y;
    float scale;
    uint laplacian;
}  IPoint_small;

	//cl_int4 width height step filter
	//x y z w

//typedef struct {
//	uint width;
//	uint height;
//	uint filter;

//}  ResponseLayer;


float getResponse(unsigned int width, unsigned int height, unsigned int row, unsigned int col, __local uint4* target, unsigned int num_target, __global float* responseBuf) {
	//int4 position = (int4)(column, row, target->index, 0);
	//return read_imagef(responseBuf, sampler, position).x;

	return responseBuf[num_target*width*height + row*width + col];
}

float getResponseScaled(unsigned int width, 
						unsigned int height, 
						unsigned int row, 
						unsigned int col, 
						__local uint4* target, 
						__local uint4* src, 
						unsigned int num_target,
						__global float* responseBuf) {

    int scale = (*target).x / (*src).x;

	//int4 position = (int4)(scale * column, scale * row, target->index, 0);
	//return read_imagef(responseBuf, sampler, position).x;
	return responseBuf[num_target*width*height + scale*row*width + scale*col];
}

unsigned int getLaplacian(unsigned int width, 
						  unsigned int height, 
						  unsigned int row, 
				   	  	  unsigned int col, 
				  	  	  __local uint4* target, 
				   	  	  __local uint4* src, 
						  unsigned int num_target,
				   	  	  __global unsigned int* lapSignBuf) {

    int scale = (*target).x / (*src).x;

	//int4 position = (int4)(scale * column, scale * row, target->index, 0);
	//return read_imagef(responseBuf, sampler, position).x;
	return lapSignBuf[num_target*width*height + scale*row*width + scale*col];
}

float BoxIntegralImage(int row, int col, int rows, int cols, __global float* imageData, unsigned int width, unsigned int height) {	
	//row, col is the upper left corner of the box
	//row+rows, col+cols is the lower right corner of the box
	
	
    // Subtract by 1 because it is inclusive
    int r1 = min(row, (int)height) - 1;
    int c1 = min(col, (int)width) - 1;
    int r2 = min(row + rows, (int)height) - 1;
    int c2 = min(col + cols, (int)width) - 1;

    float A = 0.f, B = 0.f, C = 0.f, D = 0.f;

	//Top left
    if (r1 >= 0 && c1 >= 0) {
		A = imageData[r1*width + c1];
	}
	
	//Top right
    if (r1 >= 0 && c2 >= 0) {
		B = imageData[r1*width + c2];
	}

	//Bottom left
    if (r2 >= 0 && c1 >= 0) {
		C = imageData[r2*width + c1];
	}
	
	//Bottom right
    if (r2 >= 0 && c2 >= 0) {
		D = imageData[r2*width + c2]; 
	}

    return fmax(0.0f, A - B - C + D);
}

// Non Maximal Suppression function
bool isExtremum(int r, 
				int c, 
				__local uint4* b, 
				__local uint4* m,
				__local uint4* t,
				uint num_b, uint num_m, uint num_t, 
				__global float* responseBuf,
				unsigned int width, 
				unsigned int height) {
	// bounds check
	int layerBorder = ((*t).w + 1) / (2 * (*t).z);
	if (r <= layerBorder || r >= (*t).y - layerBorder || c <= layerBorder || c >= (*t).x - layerBorder)
		return false;

	// check the candidate point in the middle layer is above thresh 
	float THRESH = 0.0004f;
	float candidate = getResponseScaled(width, height, r, c, m, t, num_m, responseBuf);
	//printf("%d\n", candidate);
	if (candidate < THRESH) 
		return false; 

	// Compare with all other pixels in a 3x3 cube
	if (getResponse(width, height, r-1, c-1, t, num_t, responseBuf) >= candidate ||
		getResponseScaled(width, height, r-1, c-1, m, t, num_m, responseBuf) >= candidate ||
		getResponseScaled(width, height, r-1, c-1, b, t, num_b, responseBuf) >= candidate) 
		return false;

	if (getResponse(width, height, r-1, c, t, num_t, responseBuf) >= candidate ||
		getResponseScaled(width, height, r-1, c, m, t, num_m, responseBuf) >= candidate ||
		getResponseScaled(width, height, r-1, c, b, t, num_b, responseBuf) >= candidate) 
		return false;

	if (getResponse(width, height, r-1, c+1, t, num_t, responseBuf) >= candidate ||
		getResponseScaled(width, height, r-1, c+1, m, t, num_m, responseBuf) >= candidate ||
		getResponseScaled(width, height, r-1, c+1, b, t, num_b, responseBuf) >= candidate) 
		return false;

	if (getResponse(width, height, r, c-1, t, num_t, responseBuf) >= candidate ||
		getResponseScaled(width, height, r, c-1, m, t, num_m, responseBuf) >= candidate ||
		getResponseScaled(width, height, r, c-1, b, t, num_b, responseBuf) >= candidate) 
		return false;

	// Do not compare the middle layer middle pixel with candidate
	if (getResponse(width, height, r, c, t, num_t, responseBuf) >= candidate ||
		getResponseScaled(width, height, r, c, b, t, num_b, responseBuf) >= candidate)
		return false;

	if (getResponse(width, height, r, c+1, t, num_t, responseBuf) >= candidate ||
		getResponseScaled(width, height, r, c+1, m, t, num_m, responseBuf) >= candidate ||
		getResponseScaled(width, height, r, c+1, b, t, num_b, responseBuf) >= candidate) 
		return false;

	if (getResponse(width, height, r+1, c-1, t, num_t, responseBuf) >= candidate ||
		getResponseScaled(width, height, r+1, c-1, m, t, num_m, responseBuf) >= candidate ||
		getResponseScaled(width, height, r+1, c-1, b, t, num_b, responseBuf) >= candidate) 
		return false;

	if (getResponse(width, height, r+1, c, t, num_t, responseBuf) >= candidate ||
		getResponseScaled(width, height, r+1, c, m, t, num_m, responseBuf) >= candidate ||
		getResponseScaled(width, height, r+1, c, b, t, num_b, responseBuf) >= candidate) 
		return false;

	if (getResponse(width, height, r+1, c+1, t, num_t, responseBuf) >= candidate ||
		getResponseScaled(width, height, r+1, c+1, m, t, num_m, responseBuf) >= candidate ||
		getResponseScaled(width, height, r+1, c+1, b, t, num_b, responseBuf) >= candidate) 
		return false;

	return true;
}




// Computes the partial derivatives in x, y, and scale of a pixel.
void deriveImage(int r, 
				 int c, 
				 __local uint4* b, 
				 __local uint4* m,
				 __local uint4* t, 
				 uint num_b, uint num_m, uint num_t, 
				 float* derivatives, 
				 __global float* responseBuf, 
				 unsigned int width, 
				 unsigned int height) {
  

  derivatives[0] = (getResponseScaled(width, height, r, c + 1, m, t, num_m, responseBuf) - getResponseScaled(width, height, r, c - 1, m, t, num_m, responseBuf)) / 2.0f;
  derivatives[1] = (getResponseScaled(width, height, r + 1, c, m, t, num_m, responseBuf) - getResponseScaled(width, height, r - 1, c, m, t, num_m, responseBuf)) / 2.0f;
  derivatives[2] = (getResponse(width, height, r, c, t, num_t, responseBuf) -getResponseScaled(width, height, r, c, b, t, num_b, responseBuf)) / 2.0f;


}


// Computes the 3D Hessian matrix for a pixel.
void buildHessian(int r, 
				  int c, 
				  __local uint4* b, 
				  __local uint4* m, 
				  __local uint4* t, 
				  uint num_b, uint num_m, uint num_t, 
				  float* hessian, 
				  __global float* responseBuf, 
				  unsigned int width, 
				  unsigned int height) {
	float v;

	v = getResponseScaled(width, height, r, c, m, t, num_m, responseBuf);
	hessian[0] = getResponseScaled(width, height, r, c + 1, m, t, num_m, responseBuf) + getResponseScaled(width, height, r, c - 1, m, t, num_m, responseBuf) - 2.f * v;
	hessian[4] = getResponseScaled(width, height, r + 1, c, m, t, num_m, responseBuf) + getResponseScaled(width, height, r - 1, c, m, t, num_m, responseBuf) - 2.f * v;
	hessian[8] = getResponse(width, height, r, c, t, num_t, responseBuf) + getResponseScaled(width, height, r, c, b, t, num_b, responseBuf) - 2.f * v;
	hessian[1] = hessian[3] = ( getResponseScaled(width, height, r + 1, c + 1, m, t, num_m, responseBuf) - getResponseScaled(width, height, r + 1, c - 1, m, t, num_m, responseBuf) - 
			getResponseScaled(width, height, r - 1, c + 1, m, t, num_m, responseBuf) + getResponseScaled(width, height, r - 1, c - 1, m, t, num_m, responseBuf) ) / 4.0f;
	hessian[2] = hessian[6] = ( getResponse(width, height, r, c + 1, t, num_t, responseBuf) - getResponse(width, height, r, c - 1, t, num_t, responseBuf) - 
			getResponseScaled(width, height, r, c + 1, b, t, num_b, responseBuf) + getResponseScaled(width, height, r, c - 1, b, t, num_b, responseBuf) ) / 4.0f;
	hessian[5] = hessian[7] = ( getResponse(width, height, r + 1, c, t, num_t, responseBuf) - getResponse(width, height, r - 1, c, t, num_t, responseBuf) - 
			getResponseScaled(width, height, r + 1, c, b, t, num_b, responseBuf) + getResponseScaled(width, height, r - 1, c, b, t, num_b, responseBuf) ) / 4.0f;
	


}


void inverseMatrix(float matrix[9]) {
	// 	0a	1b	2c
	//	3d	4e	5f
	//	6g	7h	8i
	float a, b, c, d, e, f, g, h, i;

	a = matrix[0];
	b = matrix[1];
	c = matrix[2];
	d = matrix[3];
	e = matrix[4];
	f = matrix[5];
	g = matrix[6];
	h = matrix[7];
	i = matrix[8];

	float det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h;

	float adjugate[9] = { e*i - f*h, -(b*i - c*h), b*f - c*e,
						  -(d*i - f*g), a*i - c*g, -(a*f - c*d),
						  d*h - e*g, -(a*h - b*g), a*e - b*d };

	//#pragma unroll
	for (int i = 0; i < 9; i++) {
		matrix[i] = adjugate[i]/det;
	}
}

void multiplyMatrix( float matrix[9], float column[3], float alpha, float ret[3]) {
	// alpha * (matrix * column)

	// Column
	// 0x
	// 1y
	// 2z

	// Matrix
	// 	0a	1b	2c
	//	3d	4e	5f
	//	6g	7h	8i

	// Final
	// ax + by + cz
	// dx + ey + fz
	// gx + hy + iz

	ret[0] = alpha * (matrix[0]*column[0] + matrix[1]*column[1] + matrix[2]*column[2]);
	ret[1] = alpha * (matrix[3]*column[0] + matrix[4]*column[1] + matrix[5]*column[2]);
	ret[2] = alpha * (matrix[6]*column[0] + matrix[7]*column[1] + matrix[8]*column[2]);
}




//Doing Kernel per Row. Improves Performance? Lower Overhead? image half dist and lobe size and filter size convert to local
//__attribute__((num_compute_units(1)))
//__attribute__((max_work_group_size(512)))
__kernel void buildResponseLayer (__global float* integralImage, 
								  __constant uint4* responseMap,
								  __global int* filterWidth,
								  __global int* filterBorder,
								  __global int* filterLobe,
								  __global int* filterStep,
								  __global float* normaliseFactor,
								  const unsigned int width,
								  const unsigned int height,
								  __global float* responseBuf, 
								  __global unsigned int* lapSignBuf) { 


	//copy from global to local
	/*__local int filterWidth[12];
	__local int filterBorder[12];
	__local int filterLobe[12];
	__local int filterStep[12];
	__local float normaliseFactor[12];

	event_t copy_events[5];
	copy_events[0] = async_work_group_copy(filterWidth, g_filterWidth, 12, 0);
	copy_events[1] = async_work_group_copy(filterBorder, g_filterBorder, 12, 0);
	copy_events[2] = async_work_group_copy(filterLobe, g_filterLobe, 12, 0);
	copy_events[3] = async_work_group_copy(filterStep, g_filterStep, 12, 0);
	copy_events[4] = async_work_group_copy(normaliseFactor, g_normaliseFactor, 12, 0);*/

	//for (int i = 0; i < 12; i++) { printf("========filter width: %f=======", normaliseFactor[i]);}


	float Dxx, Dyy, Dxy;
	int row, col;
	int startRow = get_local_id(0);
	int globalSize = get_global_size(0);
	//cl_int4 width height step filter
	//x y z w

	//wait_group_events(5, copy_events);

	//#pragma unroll
	for(unsigned int i = 0; i < 12; i++){
		for (unsigned int ar = startRow; ar < responseMap[i].y; ar += globalSize)
			for(unsigned int ac = 0; ac < responseMap[i].x; ac++) { 
			//if(responseMap[i].width > ac && responseMap[i].height > ar){ 

				row = ar * filterStep[i];
				col = ac * filterStep[i];


				//printf("\nRow: %d, Col: %d, %d, %d, %d", row, col, filterLobe[i], filterBorder[i], filterWidth[i]);
				//printf("%f\n", BoxIntegralImage(row - filterLobe[i] + 1, col - filterBorder[i], 2 * filterLobe[i] - 1, filterWidth[i], integralImage, width, height));
				//printf("%f\n", integralImage[row*width + col]);
				Dxx = (BoxIntegralImage(row - filterLobe[i] + 1, col - filterBorder[i], 2 * filterLobe[i] - 1, filterWidth[i], integralImage, width, height)
						- BoxIntegralImage(row - filterLobe[i] + 1, col - filterLobe[i] / 2, 2 * filterLobe[i] - 1, filterLobe[i], integralImage, width, height) * 3.f)/normaliseFactor[i];

				Dyy = (BoxIntegralImage(row - filterBorder[i], col - filterLobe[i] + 1, filterWidth[i], 2 * filterLobe[i] - 1, integralImage, width, height)
						- BoxIntegralImage(row - filterLobe[i] / 2, col - filterLobe[i] + 1, filterLobe[i], 2 * filterLobe[i] - 1, integralImage, width, height) * 3.f)/normaliseFactor[i];

				Dxy = (BoxIntegralImage(row - filterLobe[i], col + 1, filterLobe[i], filterLobe[i], integralImage, width, height)
						+ BoxIntegralImage(row + 1, col - filterLobe[i], filterLobe[i], filterLobe[i], integralImage, width, height)
						- BoxIntegralImage(row - filterLobe[i], col - filterLobe[i], filterLobe[i], filterLobe[i], integralImage, width, height)
						- BoxIntegralImage(row + 1, col + 1, filterLobe[i], filterLobe[i], integralImage, width, height))/normaliseFactor[i];
					
				//int4 pixel = (int4)(ac, ar, i, 0);
				responseBuf[i*width*height + ar*width + ac] = mad((float)Dxx, (float)Dyy, (float)- 0.81f * Dxy * Dxy);	
				//printf("%f\n", Dxx);
				lapSignBuf[i*width*height + ar*width + ac] = (Dxx + Dyy >= 0 ? 1 : 0);		
			//}	
		}
	}
}


// Each kernel computes at least 1 row and only does 1 interval
//__attribute__((num_compute_units(1)))
//__attribute__((max_work_group_size(512)))
__kernel void getIPoints(const unsigned int width, /*Width of original image in pixels*/
						 const unsigned int height,/*Height of original image in pixels*/
						 const __global uint4*  responseMap, /*Array of all response layers*/
						 __global IPoint_small*  ipts, /*Array of IPoints of size t.width * t.height */
						 __global unsigned int*  iptsIndex, /*Array of the no of IPoints in each row. t.height in size*/
						 __global float*  responseBuf, /*Determinant of hessian responses of image*/
						 __global unsigned int*  lapSignBuf) /*Laplacian signs of image*/ {

	uint filter_map[5][4] = { { 0,1,2,3 },{ 1,3,4,5 },{ 3,5,6,7 },{ 5,7,8,9 },{ 7,9,10,11 } };

	// Copies response layers to local mem for all workitems
	//cl_int4 width height step filter
	//x y z w
	unsigned int id = get_group_id(0);

	uint num_b = filter_map[id/2][id%2];
	uint num_m = filter_map[id/2][id%2 + 1];
	uint num_t = filter_map[id/2][id%2 + 2];
	
	__local uint4 b, m, t;

	event_t copy_events[3];

	copy_events[0] = async_work_group_copy(&b, &responseMap[num_b], 4, 0);
	copy_events[1] = async_work_group_copy(&m, &responseMap[num_m], 4, 0);
	copy_events[2] = async_work_group_copy(&t, &responseMap[num_t], 4, 0);
	wait_group_events(3, copy_events);
	

	//int4 b = responseMap[num_b];
	//int4 m = responseMap[num_m];
	//int4 t = responseMap[num_t];

	int tHeight = t.y;
	int tWidth = t.x;
	// Get the offsets to the actual location of the extremum
	float xi = 0.f, xr = 0.f, xc = 0.f;
	// Interpolate step
	float derivatives[3];
	float hessian[9];
	float x[3] = { 0 };
	// loop over middle response layer at density of the most 
	// sparse layer (always top), to find maxima across scale and space
	for(unsigned int r = get_local_id(0); r < tHeight; r += get_local_size(0)) { //iterate other additional rows if no. of work items per interval < height
		for (unsigned int c = 0; c < tWidth; c++) { //iterate over all cols in the row
			if (isExtremum(r, c, &b, &m, &t, num_b, num_m, num_t, responseBuf, width, height)) {
				// get the step distance between filters
				// check the middle filter is mid way between top and bottom
				unsigned int filterStep = (m.w - b.w);
				if(!(filterStep > 0 && t.w - m.w == m.w - b.w)) { 
					continue;
				}
 
				deriveImage(r, c, &b, &m, &t, num_b, num_m, num_t, derivatives, responseBuf, width, height);
				buildHessian(r, c, &b, &m, &t, num_b, num_m, num_t, hessian, responseBuf, width, height);
				inverseMatrix(hessian);
				multiplyMatrix(hessian, derivatives, -1.0f, x);

				xi = x[2];
				xr = x[1];
				xc = x[0];

				// If point is sufficiently close to the actual extremum
				if( fabs( xi ) < 0.5f  &&  fabs( xr ) < 0.5f  &&  fabs( xc ) < 0.5f ) {
					/*int index = r*width + iptsInRow[r]; // get the position to store the ipt
					ipts[index].x = (float)((c + xc) * t.step);
					ipts[index].y = (float)((r + xr) * t.step);
					ipts[index].scale = (float)((0.1333f) * (m.filter + xi*filterStep));
					ipts[index].laplacian = getLaplacian(width, height, r, c, &m, &t, lapSignBuf);

					iptsInRow[r]++; // increment the no of ipts in the row*/
					unsigned int index = atomic_add(iptsIndex, 1);
					ipts[index].x = (float)((c + xc) * t.z);
					ipts[index].y = (float)((r + xr) * t.z);
					ipts[index].scale = (float)((0.1333f) * (m.w + xi*filterStep));
					ipts[index].laplacian = getLaplacian(width, height, r, c, &m, &t, num_m, lapSignBuf);
				}
			}
		}
	}
}


