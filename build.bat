cls
nvcc main.cpp kernel.cu --use-local-env --cl-version 2012 --machine 32 -Xcompiler "/EHsc /O2 /Zi /MD" opengl32.lib GL/lib/freeglut.lib GL/lib/glew32.lib GL/lib/glew32s.lib
