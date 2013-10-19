#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

__global__ void render( uchar *output, uint width, uint height, float time )
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float) width;
	float v = y / (float) height;

	//uint4 t = make_uint4( 0 * 255, w * 255, 0 * 255, 255 );
	//output[ ( y * width + x ) * 4 + 0 ] = t.x;
	//output[ ( y * width + x ) * 4 + 1 ] = t.y;
	//output[ ( y * width + x ) * 4 + 2 ] = t.z;
	//output[ ( y * width + x ) * 4 + 3 ] = t.w;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;
	uint i = y * width + x;
	output[i*4+0] = 0;
	output[i*4+1] = w * 255;
	output[i*4+2] = w * 255;
	output[i*4+3] = 255;
}

extern "C" void render_kernel( dim3 grid, dim3 block, uchar* output, uint width, uint height, float time )
{
	render<<< grid, block>>>( output, width, height, time );
}

