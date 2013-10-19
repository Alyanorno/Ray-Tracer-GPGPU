#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

__device__ float dot( float3 a, float3 b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float length( float3 a )
{
	return sqrt( a.x * a.x + a.y * a.y + a.z * a.z );
}
__device__ float3 normalize( float3 a )
{
	float t = length( a );
	a.x = a.x / t;
	a.y = a.y / t;
	a.z = a.z / t;
	return a;
}
__device__ float3 mul( float3 a, float3 b )
{
	a.x = a.x * b.x;
	a.y = a.y * b.y;
	a.z = a.z * b.z;
	return a;
}
__device__ float3 div( float3 a, float3 b )
{
	a.x = a.x / b.x;
	a.y = a.y / b.y;
	a.z = a.z / b.z;
	return a;
}
__device__ float3 plus( float3 a, float3 b )
{
	a.x = a.x + b.x;
	a.y = a.y + b.y;
	a.z = a.z + b.z;
	return a;
}
__device__ float3 minus( float3 a, float3 b )
{
	a.x = a.x - b.x;
	a.y = a.y - b.y;
	a.z = a.z - b.z;
	return a;
}

#define NUMBER_OF_SPHERES 3
struct Sphere
{
	float3 position, color;
	float radius;
};
__device__ Sphere make_sphere( float3 position, float3 color, float radius )
{
	Sphere s;
	s.position = position;
	s.color = color;
	s.radius = radius;
	return s;
}

__device__ float3 CastRay( float3 origin, float3 direction )
{
	Sphere spheres[ NUMBER_OF_SPHERES ];
	spheres[0] = make_sphere( make_float3( 100, 0, 0 ), make_float3( 1, 0, 0 ), 100 );
	spheres[1] = make_sphere( make_float3( 0, 100, 0 ), make_float3( 0, 1, 0 ), 100 );
	spheres[2] = make_sphere( make_float3( 150, 150, 0 ), make_float3( 0, 0, 1 ), 100 );

	float max_distance = 1000;

	float3 color = make_float3( 0, 0, 0 );
	for( int i(0); i < NUMBER_OF_SPHERES; i++ )
	{
	        float3 distance = minus( origin, spheres[i].position );
	
	        float fdot = -dot( distance, direction );
	        if( fdot < 0 )
	                continue;
	        float det = fdot * fdot - dot( distance, distance ) + spheres[i].radius * spheres[i].radius;
	        if( det < 0 )
	                continue;

		float result1 = fdot - sqrt( det );
		float result2 = fdot + sqrt( det );
		float fdistance = result1 > 0 ? result1 : result2;

		if( fdistance < max_distance )
		{
			max_distance = fdistance;
			color = spheres[i].color;
		}
	}

	return color;
}

__global__ void render( uchar *output, uint width, uint height, float time )
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// calculate simple sine wave pattern
	float u = x / (float) width;
	float v = y / (float) height;
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	float3 color = make_float3( w, w, w );
	float3 origin = make_float3( 0, 0, -500 );
	float3 direction = make_float3( (int)x - (int)width/2, (int)y - (int)height/2, 0 );
	direction = normalize( minus( direction, origin ) );

	color = CastRay( origin, direction );

	uint i = y * width + x;
	output[i*4+0] = (color.x > 1 ? 1 : color.x) * 255;
	output[i*4+1] = (color.y > 1 ? 1 : color.y) * 255;
	output[i*4+2] = (color.z > 1 ? 1 : color.z) * 255;
	output[i*4+3] = 255;
}

extern "C" void render_kernel( dim3 grid, dim3 block, uchar* output, uint width, uint height, float time )
{
	render<<< grid, block>>>( output, width, height, time );
}

