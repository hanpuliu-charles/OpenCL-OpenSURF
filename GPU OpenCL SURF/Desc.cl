__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
typedef struct {
	float x;
	float y;
    float scale;
    float response;
    float orientation;
    uint laplacian;
	int descriptor_length;
	float descriptors[64];
}  IPoint;


//! lookup table for 2d gaussian (sigma = 2.5) where (0,0) is top left and (6,6) is bottom right
__constant float gauss25 [7][7] = {
  { 0.02546481f,0.02350698f,	0.01849125f,	0.01239505f,	0.00708017f,	0.00344629f,	0.00142946f},
  {0.02350698f,	0.02169968f,	0.01706957f,	0.01144208f,	0.00653582f,	0.00318132f,	0.00131956f},
  {0.01849125f,	0.01706957f,	0.01342740f,	0.00900066f,	0.00514126f,	0.00250252f,	0.00103800f},
  {0.01239505f,	0.01144208f,	0.00900066f,	0.00603332f,	0.00344629f,	0.00167749f,	0.00069579f},
  {0.00708017f,	0.00653582f,	0.00514126f,	0.00344629f,	0.00196855f,	0.00095820f,	0.00039744f},
  {0.00344629f,	0.00318132f,	0.00250252f,	0.00167749f,	0.00095820f,	0.00046640f,	0.00019346f},
  {0.00142946f,	0.00131956f,	0.00103800f,	0.00069579f,	0.00039744f,	0.00019346f,	0.00008024f}
};
#ifndef M_PI
    #define M_PI 3.14159265358979323846f
#endif

float BoxIntegralImage(int row, int col, int rows, int cols, __global float* restrict imageData, unsigned int width, unsigned int height) {	
	//row, col is the upper left corner of the box
	//row+rows, col+cols is the lower right corner of the box
	
	
    // Subtract by 1 because it is inclusive
    int r1 = min(row, (int)height) - 1;
    int c1 = min(col, (int)width) - 1;
    int r2 = min(row + rows, (int)height) - 1;
    int c2 = min(col + cols, (int)width) - 1;

    float A = 0, B = 0, C = 0, D = 0;

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

float getAngle(float x, float y){
  if(x > 0 && y >= 0)
    return atan(y/x);

  if(x < 0 && y >= 0)
    return M_PI - atan(-y/x);

  if(x < 0 && y < 0)
    return M_PI + atan(y/x);

  if(x > 0 && y < 0)
    return 2*M_PI - atan(-y/x);

  return 0;
}

float gaussianInt(int x, int y, float sig){
  return (1.0f/(2.0f*M_PI*sig*sig)) * exp( -(x*x+y*y)/(2.0f*sig*sig));
}

float gaussianFloat(float x, float y, float sig){
  return 1.0f/(2.0f*M_PI*sig*sig) * exp( -(x*x+y*y)/(2.0f*sig*sig));
}

int fRound(float flt){
  return (int) floor(flt+0.5f);
}

float haarX(int row, int column, int s, __global float* image,unsigned int width, unsigned int height){ 
	return BoxIntegralImage(row-s/2,column,s,s/2,image,width,height)-BoxIntegralImage(row-s/2,column-s/2,s,s/2,image,width,height);

}

float haarY(int row, int column, int s, __global float* image,unsigned int width, unsigned int height){ 
	return BoxIntegralImage(row, column-s/2, s/2, s, image,width,height)-BoxIntegralImage(row-s/2, column-s/2, s/2, s, image,width,height);
}


__kernel void computeDesc(__global float* restrict integralImage, 
						  __global IPoint* restrict ipts, 
						  __global unsigned int* restrict iptsCount, 
						  const unsigned int width, 
						  const unsigned int height) { 

	int localSize = get_local_size(0);
	unsigned int l_iptsCount = *iptsCount;
	
	for(int pt = get_local_id(0); pt < l_iptsCount; pt += localSize) {

	//Calculate Angle Now
	IPoint interestPoint = ipts[pt];

	//printf("\n%f, %f\n", interestPoint.x, interestPoint.y);
	//printf("Scale: %f\n", interestPoint.scale);
	//printf("Lap: %d\n", interestPoint.laplacian);

	float gauss = 0.f, scale = interestPoint.scale;
	const int s = fRound(scale), r = fRound(interestPoint.y), c = fRound(interestPoint.x);
	float resX[109];
	float resY[109];
	float angle[109];
	const int id[] = {6,5,4,3,2,1,0,1,2,3,4,5,6};

	int idx = 0;
	// calculate haar responses for points within radius of 6*scale
	#pragma unroll 1
	#pragma loop_coalesce
	for(int i = -6; i <= 6; ++i) {
		#pragma unroll 1
		for(int j = -6; j <= 6; ++j) {
			if(i*i + j*j < 36) { 
				gauss=(float)(gauss25[id[i+6]][id[j+6]]);
				resX[idx]=gauss*haarX(r+j*s,c+i*s,4*s,integralImage,width,height);
				resY[idx]=gauss*haarY(r+j*s,c+i*s,4*s,integralImage,width,height);
				angle[idx]= getAngle(resX[idx], resY[idx]);
				//printf("\n%d, %f", gid, resX[idx]);
				++idx;
			}
		}
	}

	// calculate the dominant direction 
	float sumX=0.f, sumY=0.f;
	float maxVal=0.f, orientation = 0.f;
	float ang1=0.f, ang2=0.f;

	// loop slides pi/3 window around feature point
	#pragma unroll 1
	//#pragma loop_coalesce
	for(ang1 = 0; ang1 < 2*M_PI;  ang1+=0.15f) {
		ang2 = ( ang1+M_PI/3.0f > 2*M_PI ? ang1-5.0f*M_PI/3.0f : ang1+M_PI/3.0f);
		sumX = sumY = 0.f;
		#pragma unroll 1
		for(unsigned int k = 0; k < 109; ++k) {

			// get angle from the x-axis of the sample point
			float ang = angle[k];

			// determine whether the point is within the window
			if (ang1 < ang2 && ang1 < ang && ang < ang2) { 
				sumX+=resX[k];
				sumY+=resY[k];
			} else if(ang2 < ang1 && ((ang > 0 && ang < ang2) || (ang > ang1 && ang < 2*M_PI) )){
				sumX+=resX[k]; 
				sumY+=resY[k]; 
			}
		}
		// if the vector produced from this window is longer than all 
		// previous vectors then this forms the new dominant direction
		if (sumX*sumX + sumY*sumY > maxVal) {
		  // store largest orientation
		  maxVal = sumX*sumX + sumY*sumY;
		  orientation = getAngle(sumX, sumY);
		}
	}

	//printf("%f \n", orientation);
	interestPoint.orientation=orientation;
	//ipts[pt].orientation=orientation;

	//calculate SURF Descriptor Now
	int y, x, sample_x, sample_y, count=0;
	int i = 0, ix = 0, j = 0, jx = 0, xs = 0, ys = 0;
	float* restrict desc;
	float dx, dy, mdx, mdy, co, si;
	float gauss_s1 = 0.f, gauss_s2 = 0.f;
	float rx = 0.f, ry = 0.f, rrx = 0.f, rry = 0.f, len = 0.f;
	float cx = -0.5f, cy = 0.f;
	x = fRound(interestPoint.x);
	y = fRound(interestPoint.y); 
	desc=interestPoint.descriptors;
	co=cos(orientation);
	si=sin(orientation);
	i=-8;
	//Here we start calculate
	while(i<12){ 
		j=-8;
		i-=4;
		cx+=1.f;
		cy=-0.5f;
		while(j<12){ 
			dx=dy=mdx=mdy=0.f;
			cy+=1.f;
			j-=4;
			ix=i+5;
			jx=j+5;
			xs = fRound(x + ( -jx*scale*si + ix*scale*co));
			ys = fRound(y + ( jx*scale*co + ix*scale*si));
			#pragma unroll 1
			#pragma loop_coalesce
			for(int k=i;k<i+9;k++){ 
				#pragma unroll 1
				for(int l=j;l<j+9;l++){ 
					sample_x = fRound(x + (-l*scale*si + k*scale*co));
					sample_y = fRound(y + ( l*scale*co + k*scale*si));
					gauss_s1 = gaussianFloat(xs-sample_x,ys-sample_y,2.5f*scale);
					rx = haarX(sample_y, sample_x, 2*fRound(scale),integralImage,width,height);
					ry = haarY(sample_y, sample_x, 2*fRound(scale),integralImage,width,height);
					rrx = gauss_s1*(-rx*si + ry*co);
					rry = gauss_s1*(rx*co + ry*si);
					dx += rrx;
					dy += rry;
					mdx += fabs(rrx);
					mdy += fabs(rry);
				}
			
			}
			gauss_s2 = gaussianFloat(cx-2.0f,cy-2.0f,1.5f);

			desc[count++] = dx*gauss_s2;
			desc[count++] = dy*gauss_s2;
			desc[count++] = mdx*gauss_s2;
			desc[count++] = mdy*gauss_s2;

			len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy) * gauss_s2*gauss_s2;
	
			j += 9;


		}
		i+=9;
	
	}
	len=sqrt(len);
	
	#pragma unroll 1
	for(int i=0;i<64;i++){ 
		desc[i]/=len;
		interestPoint.descriptors[i]=desc[i];
	}
	ipts[pt] = interestPoint;
	//ipts[pt].orientation = (pt%2 == 0 ? (0.0f) : (1453.3249233f));
	//printf("%f ", ipts[pt].orientation);
	//interestPoint.descriptors=desc;


	}
}
