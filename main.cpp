#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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

void cleanup();

// rendering callbacks
void display();
void keyboard( unsigned char key, int x, int y );
void mouse( int button, int state, int x, int y );
void motion( int x, int y );
void reshape( int x, int y );
void timerEvent( int value );

extern "C" void render_kernel( dim3 grid, dim3 block, uchar* output, uint width, uint height, float time );

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

