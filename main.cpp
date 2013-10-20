#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <Windows.h>

#include "GL/include/glew.h"
#include "GL/include/freeglut.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector_types.h>

typedef unsigned int uint;
typedef unsigned char uchar;
unsigned int width = 512, height = 512;

dim3 block( 16, 16, 1 );
dim3 grid( width / block.x, height / block.y );

GLuint pbo; // pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource;

uchar *output = NULL;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;

float time;

// callbacks
void cleanup();
void display();
void keyboard( unsigned char key, int x, int y );
void mouse( int button, int state, int x, int y );
void motion( int x, int y );
void reshape( int x, int y );
void timerEvent( int value );

extern "C" void render_kernel( dim3 grid, dim3 block, uchar* output, uint width, uint height, float time );

struct Model
{
	int _number;
	std::vector<float> _vertexs, _normals, _textureCoordinates;

	void Set( int number )
	{
		_number = number;
		_vertexs.resize( number * 3 );
		_normals.resize( number * 3 );
		_textureCoordinates.resize( number * 2 );
	}

	void LoadObj( std::string name );
} model;


int main(int argc, char **argv)
{
	time = 0.f;

	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
	glutInitWindowSize( width, height );
	glutCreateWindow( "Ray Tracer" );
	glutDisplayFunc( display );
	glutKeyboardFunc( keyboard );
	glutMotionFunc( motion );
	glutReshapeFunc(reshape);
	glutTimerFunc( 10, timerEvent, 0 );

	glewInit();

	cudaGLSetGLDevice(0);

	glGenBuffers( 1, &pbo );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
	glBufferData( GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer( &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard );

	model.LoadObj( "bth.obj" );

	atexit(cleanup);
	glutMainLoop();
}

void display()
{
	// map PBO to get CUDA device pointer
	cudaGraphicsMapResources( 1, &cuda_pbo_resource, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer( (void **)&output, &num_bytes, cuda_pbo_resource );

	// call CUDA kernel, writing results to PBO
	render_kernel( grid, block, output, width, height, time );

	cudaGraphicsUnmapResources( 1, &cuda_pbo_resource, 0 );

	glClear( GL_COLOR_BUFFER_BIT );

	// draw image from PBO
	glDisable( GL_DEPTH_TEST );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
	glDrawPixels( width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

	glutSwapBuffers();
	glutReportErrors();

	time += 0.01f;
}

void timerEvent(int value)
{
	glutPostRedisplay();
	glutTimerFunc( 10, timerEvent,0 );
}

void cleanup()
{
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

	cudaGraphicsUnregisterResource(cuda_pbo_resource);
	glDeleteBuffers(1, &pbo);
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
		case (27) :
			exit(EXIT_SUCCESS);
			break;
	}
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1<<button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void reshape(int x, int y)
{
	width = x;
	height = y;
	grid = dim3( width / block.x, height / block.y );

	glViewport(0, 0, width, height);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

	cudaGraphicsUnregisterResource(cuda_pbo_resource);
	glDeleteBuffers(1, &pbo);

	glGenBuffers( 1, &pbo );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo );
	glBufferData( GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer( &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard );
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		//translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void Tokenize( std::string line, std::vector< std::string >& result, char delim = ' ' )
{
	std::stringstream ss( line );
	while( getline( ss, line, delim ) )
		result.push_back( line );
}
void Model::LoadObj( std::string name )
{
	std::fstream in;
	in.open( name.c_str(), std::ios::in );
	if( !in.is_open() )
		;

	std::vector<float> vertexs, normals, textureCoordinates;
	std::vector<unsigned int> faces;

	while( !in.eof() )
	{
		std::string line;
		getline( in, line );

		if( !line.size() )
			continue;

		if( line[0] == 'v' ) {
			std::vector< std::string > t;
			Tokenize( line, t );
			if( line[1] == 't' )
				// Texture Coordinate
				for( int i(1); i < t.size(); i++ )
					textureCoordinates.push_back( stof( t[i] ) );
			else if( line[1] == 'n' )
				// Normal
				for( int i(1); i < t.size(); i++ )
					normals.push_back( stof( t[i] ) );
			else
				// Vertex
				for( int i(1); i < t.size(); i++ )
					vertexs.push_back( stof( t[i] ) );
		} else if( line[0] == 'f' ) {
			// Face
			std::vector< std::string > t;
			Tokenize( line, t );
			for( int i(1); i < t.size(); i++ ) {
				std::vector< std::string > y;
				Tokenize( t[i], y, '/' );
				if( y.size() != 3 ) //only supports triangles
					;
				for( int j(0); j < y.size(); j++ )
					faces.push_back( stoi( y[j] ) );
			}
		}
	}

	in.close();

	Set( faces.size() / 3 );
	for( int i(0); i < faces.size() / 3; i++ )
	{
		for( int j(0); j < 3; j++ )
		{
			_vertexs[ i * 3 + j ] = vertexs[ ( faces[ i * 3 + 0 ] - 1 ) * 3 + j ];
			_normals[ i * 3 + j ] = normals[ ( faces[ i * 3 + 2 ] - 1 ) * 3 + j ];
		}
		for( int j(0); j < 2; j++ )
			_textureCoordinates[ i * 2 + j ] = textureCoordinates[ ( faces[ i * 3 + 1 ] - 1 ) * 2 + j ];
	}
}

