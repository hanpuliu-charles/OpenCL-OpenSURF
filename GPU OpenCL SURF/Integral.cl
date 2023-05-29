
__kernel void CreateIntegralImage(const unsigned int width, const unsigned int height, __global float *imageData, __global float *result) { 
	float rs = 0.0f;

	for (int x = 0; x < width; x++) {
		rs += imageData[x];
		result[x] = rs;
	}

	//barrier(CLK_LOCAL_MEM_FENCE); // to be safe i think

	// remaining cells are sum above and to the left
	for (int y = 1; y < height; y++) { // iterate over each row
		rs = 0.0f; // stores accumulative sum of row data
		for (int x = 0; x < width; x++) { // iterate over each item in the row
			rs += imageData[y*width + x]; // accumulate row sum
			result[y*width + x] = rs + result[(y-1)*width + x]; // add the integral value of the pixel above
		}
	}

	/*for (int y = 0; y < height; y++) { 
		for (int x = 0; x < width; x++) { 
			printf("\nPoint: %d %d, value %f",x,y,result[y*width + x]);
		}
	}*/

}

/*
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void CreateIntegralImage(__read_only image2d_t imageData, __read_write image2d_t result, __global float *buffer) { 
	float rs = 0.0f;
	int width = get_image_width(imageData);
	int height = get_image_height(imageData);

	for (int x = 0; x < width; x++) {
		int2 position = {x, 0};
		rs += read_imagef(imageData, sampler, position).x;
		buffer[x] = rs;
	}


	// remaining cells are sum above and to the left
	for (int y = 1; y < height; y++) { // iterate over each row
		rs = 0.0f; // stores accumulative sum of row data
		for (int x = 0; x < width; x++) { // iterate over each item in the row
			int2 position = {x, y};
			rs += read_imagef(imageData, sampler, position).x; // accumulate row sum

			//int2 posAbove = {x, y - 1};
			//float above = read_imagef(result, sampler, posAbove).x; // get the integral value of the pixel above
			buffer[y*width + x] = rs + buffer[(y-1) * width + x];
			//write_imagef(result, position, rs + above); // write the final value into the result
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE); // to be safe i think

	for (int y = 0; y < height; y++) { 
		for (int x = 0; x < width; x++) { 
			int2 position = {x, y};
			write_imagef(result, position, buffer[y*width + x]);
			printf("\nPoint: %d %d, value %f",position[0],position[1],buffer[y*width + x]);
		}
	}
	for (int y = 0; y < height; y++) { 
		for (int x = 0; x < width; x++) { 
			int2 position = {x, y};
			float4 temp=read_imagef(result, position);
			printf("\nPoint: %d %d, value %f",position[0],position[1],temp[0]);
		}
	}
	

} */
