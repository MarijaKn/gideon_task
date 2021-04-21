# gideon_task

REQUIREMENTS

OpenCV, CUDA (version 10.2)

COMPILING

Position yourself to gideon_task directory and run:

$ cmake .

If cmake runs succesfully, run:

$ make

If make passes succesfully, run executable:

$ ./ImageProcessing

The program should print time measurements to terminal/command line and should save images (the result of each conversion).

POSSIBLE COMPILING ISSUES

I had issues compiling the project locally due to my machine's failure to locate CUDA and CUDA compiler.
If you come across error message complaining about CUDA compiler being broken, run this:

$ export CUDACXX=/usr/local/cuda-10.2/bin/nvcc

Then delete CMakeCache.txt and run cmake . again.

In case of cmake error complaining about "Could NOT find CUDA", run:

$ export CUDA_BIN_PATH=/usr/local/cuda-10.2/

Then again, delete CMakeCache.txt and run cmake . again.

After that, cmake should work.
