#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>

typedef unsigned int  uint;
typedef unsigned char uchar;


__device__ uint seed;
__device__ int random()
{
	seed = seed * 0x343FD + 0x269EC3; 
	return seed >> 16 & 0x7FFF;
}
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
__device__ float flength( float a )
{
	return a > 0 ? a: -a;
}
__device__ float calc_distance( float3 a, float3 b )
{
	a.x = a.x - b.x;
	a.y = a.y - b.y;
	a.z = a.z - b.z;
	return length( a );
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


struct Material
{
	float reflection, ambient, diffuse, specular;
} __device__ material;
__device__ Material make_material( float reflection, float ambient, float diffuse, float specular )
{
	Material m;
	m.reflection = reflection;
	m.ambient = ambient;
	m.diffuse = diffuse;
	m.specular = specular;
	return m;
}

struct Model
{
	int number;
	float *vertexs, *normals, *textureCoordinates;
	float3 position;
} __device__ model;
__device__ Model make_model( int number, float* vertexs, float* normals, float* textureCoordinates, float3 position )
{
	Model m;
	m.number = number;
	m.vertexs = vertexs;
	m.normals = normals;
	m.textureCoordinates = textureCoordinates;
	m.position = position;
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

#define NUMBER_OF_PARTICLES 100
#define PARTICLE_RADIUS 5.f
struct Particle
{
	float3 position, direction;
} __device__ particles[ NUMBER_OF_PARTICLES ];

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
		for( int l(0); l < model.number; )
		{
			float3 point1 = plus( make_float3( model.vertexs[l*3+0], model.vertexs[l*3+1], model.vertexs[l*3+2]), model.position );
			++l;
			float3 point2 = plus( make_float3( model.vertexs[l*3+0], model.vertexs[l*3+1], model.vertexs[l*3+2]), model.position );
			++l;
			float3 point3 = plus( make_float3( model.vertexs[l*3+0], model.vertexs[l*3+1], model.vertexs[l*3+2]), model.position );
			++l;
			float3 e1 = minus( point2, point1 );
			float3 e2 = minus( point3, point1 );
			float3 s = minus( origin, point1 );
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
		color = plus( color, fmul( fmul( material_color, fdot * material.diffuse ), light_strength ) );

		// Specular
		float3 reflection = minus( direction, fmul( normal, (2 * dot( normal, direction ) ) ) );
		fdot = dot( reflection, direction );
		if( fdot > 0 )
			color = plus( color, fmul( fmul( material_color, powf( fdot, 20 ) * material.specular ), light_strength ) );
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
		bool hit_particle = false;
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
		for( int l(0); l < model.number; )
		{
			float3 point1 = plus( make_float3( model.vertexs[l*3+0], model.vertexs[l*3+1], model.vertexs[l*3+2]), model.position );
			++l;
			float3 point2 = plus( make_float3( model.vertexs[l*3+0], model.vertexs[l*3+1], model.vertexs[l*3+2]), model.position );
			++l;
			float3 point3 = plus( make_float3( model.vertexs[l*3+0], model.vertexs[l*3+1], model.vertexs[l*3+2]), model.position );
			++l;
			float3 e1 = minus( point2, point1 );
			float3 e2 = minus( point3, point1 );
			float3 s = minus( origin, point1 );
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
				color = make_float3( 0.8, 1, 1 );
				normal = normalize( cross( e2, e1 ) );
			}
		}
		for( int l(0); l < NUMBER_OF_PARTICLES; l++ )
		{
			float3 distance = minus( origin, particles[l].position );

			float fdot = -dot( distance, direction );
			if( fdot < 0 )
				continue;
			float det = fdot * fdot - dot( distance, distance ) + PARTICLE_RADIUS * PARTICLE_RADIUS;
			if( det < 0 )
				continue;

			float result1 = fdot - sqrt( det );
			float result2 = fdot + sqrt( det );
			float fdistance = result1 > 0 ? result1 : result2;

			if( fdistance < max_distance )
			{
				max_distance = fdistance;
				color = make_float3( 0, 0.5f, 1 );
				hit_particle = true;
			}
		}

		bool hit = false;

		if( !hit_particle )
		{
			hit = color.x + color.y + color.z > 0 ? true: false;

			origin = plus( origin, fmul( direction, max_distance - 0.1f ) );
			direction = minus( direction, fmul( normal, (2 * dot( normal, direction ) ) ) );

			if( hit )
				color = LightRay( origin, normal, color );
		}

		if( i == 0 )
		{
			result = color;
			reflection = material.reflection;
		}
		else
		{
			reflection = reflection * material.reflection;
			result = plus( result, fmul( color, reflection ) );
		}

		if( !hit )
			break;
	}

	return result;
}

// calculate simple sine wave pattern
__device__ float getWavePattern( float frequency, float time )
{
	return sinf(frequency + time) * cosf(frequency + time);
}

__global__ void render( uchar *output, uint width, uint height, float time )
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	lights[0].position.x = 100 + getWavePattern( 0.2, time ) * 200;
	lights[0].position.y = -100 + getWavePattern( 0.4, time ) * 200;

	lights[1].position.x = -100 + getWavePattern( 0.2, time ) * 200;
	lights[1].position.y = -100 + getWavePattern( 0.2, time ) * 200;
	lights[1].position.z = -200 + getWavePattern( 0.5, time ) * 300;

	float3 origin = make_float3( 0, 0, -500 );
	float3 direction;
	float3 color, color1, color2, color3, color4;

	float diff_x = 0.5f;
	float diff_y = 0.5f;

/*	direction = make_float3( (int)x - (int)width/2, (int)y - (int)height/2, 0 );
	direction = normalize( minus( direction, origin ) );
	color = CastRay( origin, direction ); */


	direction = make_float3( (int)x - (int)width/2 + diff_x, (int)y - (int)height/2 + diff_y, 0 );
	direction = normalize( minus( direction, origin ) );
	color1 = CastRay( origin, direction );

	direction = make_float3( (int)x - (int)width/2 + diff_x, (int)y - (int)height/2 - diff_y, 0 );
	direction = normalize( minus( direction, origin ) );
	color2 = CastRay( origin, direction );

	direction = make_float3( (int)x - (int)width/2 - diff_x, (int)y - (int)height/2 + diff_y, 0 );
	direction = normalize( minus( direction, origin ) );
	color3 = CastRay( origin, direction );

	direction = make_float3( (int)x - (int)width/2 - diff_x, (int)y - (int)height/2 - diff_y, 0 );
	direction = normalize( minus( direction, origin ) );
	color4 = CastRay( origin, direction );

	uint i = y * width + x;
	color = fmul( plus( color1, plus( color2, plus( color3, color4 ) ) ), 1.f/4.f );
	output[i*4+0] = (color.x > 1 ? 1 : color.x) * 255;
	output[i*4+1] = (color.y > 1 ? 1 : color.y) * 255;
	output[i*4+2] = (color.z > 1 ? 1 : color.z) * 255;
	output[i*4+3] = 255;
}

__global__ void update()
{
	uint i = blockIdx.x;

	// Alignment
	float3 alignment = make_float3( 0, 0, 0 );
	uint alignment_count(0);

	// Cohesion
	float3 cohesion = make_float3( 0, 0, 0 );
	uint cohesion_count(0);

	// Separation
	float3 separation = make_float3( 0, 0, 0 );
	uint separation_count(0);

	/*
	int l = i - 50;
	l = l < 0 ? 0: l;
	int stop = i + 50;
	stop = stop > _size ? _size: stop;
	for(; l < stop; l++ ) {}
	*/

	for( int l(0); l < NUMBER_OF_PARTICLES; l++ )
	{
		if( i == l )
			continue;

		float distance = calc_distance( particles[i].position, particles[l].position );

		// Alignment
		if( distance < 20 )
		{
			alignment = plus( alignment, particles[l].direction );
			++alignment_count;
		}

		// Cohesion
		if( distance < 100 )
		{
			cohesion = plus( cohesion, particles[l].direction );
			++cohesion_count;
		}

		// Separation
		if( distance < 7 )
		{
			separation = plus( separation, minus( particles[l].position, particles[i].position ) );
			++separation_count;
		}
	}

	// Alignment
	if( alignment_count )
		particles[i].direction = plus( particles[i].direction, fmul( normalize( fmul( alignment, 1/(float)alignment_count ) ), 0.005 ) );

	// Cohesion
	if( cohesion_count )
	{
		float3 t = fmul( cohesion, 1/(float)cohesion_count );
		t = minus( t, particles[i].position );
		if( dot( t, t ) > 0.001f )
			particles[i].direction = plus( particles[i].direction, fmul( normalize( t ), 0.0005 ) );
	}

	// Separation
	if( separation_count )
	{
		float3 t = fmul( separation, -1/(float)separation_count );
		if( dot( t, t ) > 0.001f )
			particles[i].direction = plus( particles[i].direction, fmul( normalize( t ), 0.05 ) );
	}

	// Gravity
	float3 t = plus( particles[i].position, make_float3( 0, 0, 100 ) );
	float distance = dot( t, t ) / 20000;
	particles[i].direction = plus( particles[i].direction, fmul( normalize( t ), distance * distance * -0.005f ) );

	particles[i].direction = normalize( particles[i].direction );
	particles[i].position = plus( particles[i].position, fmul( particles[i].direction, 2.5 ) );
}

__global__ void init( int number, float* vertexs, float* normals, float* textureCoordinates)
{
	material = make_material( 0.8f, 0.2f, 0.4f, 0.4f );

	model = make_model( number, vertexs, normals, textureCoordinates, make_float3( 200, -50, -50 ) );

	spheres[0] = make_sphere( make_float3( 100, 0, 0 ), make_float3( 1, 0, 0 ), 100 );
	spheres[1] = make_sphere( make_float3( 0, 100, 0 ), make_float3( 0, 1, 0 ), 100 );
	spheres[2] = make_sphere( make_float3( 150, 150, 0 ), make_float3( 0, 0, 1 ), 100 );

	triangles[0] = make_triangle( make_float3( 100, 0, -100 ), make_float3( -100, 0, -100 ), make_float3( 120, 200, 0 ), make_float3( 0.5, 0.5, 0.5 ) );

	for( int i(0); i < NUMBER_OF_PARTICLES; i++ )
	{
		particles[i].position = make_float3( (i % 10) * 20 - 100, (i / 10) * 20 - 100, (i % 10) - 100 );
		particles[i].direction = normalize( make_float3( random(), random(), random() ) );
	}

	lights[0] = make_light( make_float3( 100, -100, -200 ), 1000 );
	lights[1] = make_light( make_float3( -200, -200, -200 ), 600 );
}

extern "C"
{
	void render_kernel( dim3 grid, dim3 block, uchar* output, uint width, uint height, float time )
	{
		render<<< grid, block >>>( output, width, height, time );
	}

	void init_kernel( int number, float* vertexs, float* normals, float* textureCoordinates)
	{
		init<<< dim3(1), dim3(1) >>>( number, vertexs, normals, textureCoordinates);
	}

	void update_kernel( float time )
	{
		update<<< dim3( NUMBER_OF_PARTICLES ), dim3( 1 ) >>>();
	}
}

