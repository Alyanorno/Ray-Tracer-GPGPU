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
__device__ float3 cross( float3 a, float3 b )
{
	a.x = a.y * b.z - a.z * b.y;
	a.y = a.z * b.x - a.x * b.z;
	a.z = a.x * b.y - a.y * b.x;
	return a;
}
__device__ float determinant( float3 a, float3 b, float3 c )
{
	return	a.x * b.y * c.z + 
		b.x * c.y * a.z + 
		c.x * a.y * b.z -
		c.x * b.y * c.x -
		b.x * a.y * c.z -
		a.x * c.y * b.z;
}
__device__ float length( float3 a )
{
	return sqrt( a.x * a.x + a.y * a.y + a.z * a.z );
}
__device__ float3 normalize( float3 a )
{
	float t = length( a );
	if( t == 0 )
		return a;
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


struct Model
{
	int number;
	float *vertexs, *normals, *textureCoordinates;
} __device__ model;
__device__ Model make_model( int number, float* vertexs, float* normals, float* textureCoordinates )
{
	Model m;
	m.number = number;
	m.vertexs = vertexs;
	m.normals = normals;
	m.textureCoordinates = textureCoordinates;
	return m;
}

#define NUMBER_OF_SPHERES 3
struct Sphere
{
	float3 position, color;
	float radius;
} __device__ spheres[ NUMBER_OF_SPHERES ];
__device__ Sphere make_sphere( float3 position, float3 color, float radius )
{
	Sphere s;
	s.position = position;
	s.color = color;
	s.radius = radius;
	return s;
}

#define NUMBER_OF_TRIANGLES 1
struct Triangle
{
	float3 point1, point2, point3, color;
} __device__ triangles[ NUMBER_OF_TRIANGLES];
__device__ Triangle make_triangle( float3 point1, float3 point2, float3 point3, float3 color )
{
	Triangle t;
	t.point1 = point1;
	t.point2 = point2;
	t.point3 = point3;
	t.color = color;
	return t;
}

#define NUMBER_OF_LIGHTS 2
struct Light
{
	float3 position;
	float radius;
} __device__ lights[ NUMBER_OF_LIGHTS ];
__device__ Light make_light( float3 position, float radius )
{
	Light l;
	l.position = position;
	l.radius = radius;
	return l;
}

__device__ bool IsNaN( float3 a )
{
	if( length(a) <= 0 )
		return false;
	else if( length(a) > 0 )
		return false;
	else
		return true;
}
__device__ float3 LightRay( float3 origin, float3 normal, float3 material_color )
{
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
		if( fdot <= 0 )
			continue;

		//Check for collision
		bool collision = false;
		for( int l(0); l < NUMBER_OF_SPHERES; l++ )
		{
			float3 distance = minus( origin, spheres[l].position );

			float ftdot = -dot( distance, direction );
			if( ftdot < 0 )
				continue;
			float det = ftdot * ftdot - dot( distance, distance ) + spheres[l].radius * spheres[l].radius;
			if( det < 0 )
				continue;
			collision = true;
			break;
		}
		for( int l(0); l < NUMBER_OF_TRIANGLES; l++ )
		{
			float3 e1 = minus( triangles[l].point2, triangles[l].point1 );
			float3 e2 = minus( triangles[l].point3, triangles[l].point1 );
			float3 s = minus( origin, triangles[l].point1 );
			float3 d = direction;

			float t = 1 / determinant( fmul( d, -1 ), e1, e2 );
			float3 result = make_float3(
				determinant( s, e1, e2 ),
				determinant( fmul( d, -1 ), s, e2 ),
				determinant( fmul( d, -1 ), e1, s ) );
			result = fmul( result, t );

			if( result.y < 0 || result.z < 0 || result.y + result.z > 1 )
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
	float ambient = 0.2;
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
		for( int l(0); l < NUMBER_OF_TRIANGLES; l++ )
		{
			float3 e1 = minus( triangles[l].point2, triangles[l].point1 );
			float3 e2 = minus( triangles[l].point3, triangles[l].point1 );
			float3 s = minus( origin, triangles[l].point1 );
			float3 d = direction;

			float t = 1 / determinant( fmul( d, -1 ), e1, e2 );
			float3 result = make_float3(
				determinant( s, e1, e2 ),
				determinant( fmul( d, -1 ), s, e2 ),
				determinant( fmul( d, -1 ), e1, s ) );
			result = fmul( result, t );

			if( result.y < 0 || result.z < 0 || result.y + result.z > 1 )
				continue;
			else if( result.x < max_distance )
			{
				max_distance = result.x;
				color = triangles[l].color;
				normal = normalize( cross( e2, e1 ) );
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
			reflection = 0.8f;
		}
		else
		{
			reflection = reflection * 0.8f;
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
	render<<< grid, block >>>( output, width, height, time );
}

__global__ void init( int number, float* vertexs, float* normals, float* textureCoordinates)
{
	spheres[0] = make_sphere( make_float3( 100, 0, 0 ), make_float3( 1, 0, 0 ), 100 );
	spheres[1] = make_sphere( make_float3( 0, 100, 0 ), make_float3( 0, 1, 0 ), 100 );
	spheres[2] = make_sphere( make_float3( 150, 150, 0 ), make_float3( 0, 0, 1 ), 100 );

	triangles[0] = make_triangle( make_float3( 100, 0, -100 ), make_float3( -100, 0, -100 ), make_float3( 120, 200, 0 ), make_float3( 0.5, 0.5, 0.5 ) );
	
	lights[0] = make_light( make_float3( 100, -100, -200 ), 1000 );
	lights[1] = make_light( make_float3( -200, -200, -200 ), 800 );

	model = make_model( number, vertexs, normals, textureCoordinates );
}

extern "C" void init_kernel( int number, float* vertexs, float* normals, float* textureCoordinates)
{
	init<<< dim3(1), dim3(1) >>>( number, vertexs, normals, textureCoordinates);
}

