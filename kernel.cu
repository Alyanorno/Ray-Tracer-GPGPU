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
__device__ float3 fmul( float3 a, float b )
{
	a.x = a.x * b;
	a.y = a.y * b;
	a.z = a.z * b;
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

#define NUMBER_OF_LIGHTS 2
struct Light
{
	float3 position;
	float radius;
};
__device__ Light make_light( float3 position, float radius )
{
	Light l;
	l.position = position;
	l.radius = radius;
	return l;
}

__device__ float3 LightRay( float3 origin, float3 normal, float3 material_color )
{
	Light lights[ NUMBER_OF_LIGHTS ];
	lights[0] = make_light( make_float3( 100, -100, -200 ), 1000 );
	lights[1] = make_light( make_float3( -200, -200, -200 ), 800 );

	float3 color = make_float3( 0, 0, 0 );
	for( int i(0); i < NUMBER_OF_LIGHTS; i++ )
	{
		float fdistance = length( minus( lights[i].position, origin ) );
		if( fdistance == 0 )
			continue;
		if( fdistance >= lights[i].radius )
			continue;
		float light_strength = 1 - fdistance / lights[i].radius;
		light_strength = light_strength > 1 ? 1: light_strength;

		float3 direction = normalize( minus( lights[i].position, origin ) );

		float fdot = dot( normal, direction );
		if( fdot < 0 )
			continue;

		//Check for collision
		bool collision = false;
		Sphere spheres[ NUMBER_OF_SPHERES ];
		spheres[0] = make_sphere( make_float3( 100, 0, 0 ), make_float3( 1, 0, 0 ), 100 );
		spheres[1] = make_sphere( make_float3( 0, 100, 0 ), make_float3( 0, 1, 0 ), 100 );
		spheres[2] = make_sphere( make_float3( 150, 150, 0 ), make_float3( 0, 0, 1 ), 100 );
		for( int l(0); l < NUMBER_OF_SPHERES; l++ )
		{
			float3 distance = minus( origin, spheres[l].position );

			float fdot = -dot( distance, direction );
			if( fdot < 0 )
				continue;
			float det = fdot * fdot - dot( distance, distance ) + spheres[l].radius * spheres[l].radius;
			if( det < 0 )
				continue;
			collision = true;
			break;
		}
		if( collision )
			continue;

		// Diffuse
		float diffuse = 0.4;
		color = plus( color, fmul( fmul( material_color, fdot * diffuse ), light_strength ) );

		// Specular
		float3 reflection = minus( direction, fmul( normal, (2 * dot( normal, direction ) ) ) );
		fdot = dot( reflection, direction );
		if( fdot > 0 )
		{
			float specular = 0.4;
			color = plus( color, fmul( fmul( material_color, powf( fdot, 20 ) * specular ), light_strength ) );
		}
	}

	// Ambient
	float ambient = 0.1;
	color = plus( color, fmul( material_color, ambient ) );

	return color;
}

__device__ float3 CastRay( float3 origin, float3 direction )
{
	float3 result = make_float3( 0, 0, 0 );
	uint max_deapth = 5;
	float reflection;
	for( uint i(0); i < max_deapth; i++ )
	{
		Sphere spheres[ NUMBER_OF_SPHERES ];
		spheres[0] = make_sphere( make_float3( 100, 0, 0 ), make_float3( 1, 0, 0 ), 100 );
		spheres[1] = make_sphere( make_float3( 0, 100, 0 ), make_float3( 0, 1, 0 ), 100 );
		spheres[2] = make_sphere( make_float3( 150, 150, 0 ), make_float3( 0, 0, 1 ), 100 );

		float max_distance = 1000;

		float3 color = make_float3( 0, 0, 0 );
		float3 normal;
		for( int l(0); l < NUMBER_OF_SPHERES; l++ )
		{
			float3 distance = minus( origin, spheres[l].position );

			float fdot = -dot( distance, direction );
			if( fdot < 0 )
				continue;
			float det = fdot * fdot - dot( distance, distance ) + spheres[l].radius * spheres[l].radius;
			if( det < 0 )
				continue;

			float result1 = fdot - sqrt( det );
			float result2 = fdot + sqrt( det );
			float fdistance = result1 > 0 ? result1 : result2;

			if( fdistance < max_distance )
			{
				max_distance = fdistance;
				color = spheres[l].color;
				normal = normalize( fmul( minus( plus( origin, fmul( direction, max_distance ) ), spheres[l].position ), spheres[l].radius ) );
			}
		}
	
		bool hit = color.x + color.y + color.z > 0 ? true: false;

		origin = plus( origin, fmul( direction, max_distance - 0.1f ) );
		direction = minus( direction, fmul( normal, (2 * dot( normal, direction ) ) ) );

		if( hit )
			color = LightRay( origin, normal, color );

		if( i == 0 )
		{
			result = color;
			reflection = 0.5f;
		}
		else
		{
			reflection = reflection * 0.5f;
			result = plus( result, fmul( color, reflection ) );
		}

		if( !hit )
			break;
	}

	return result;
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

	float3 origin = make_float3( 0, 0, -500 );
	float3 direction = make_float3( (int)x - (int)width/2, (int)y - (int)height/2, 0 );
	direction = normalize( minus( direction, origin ) );

	float3 color = CastRay( origin, direction );

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

