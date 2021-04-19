cmake_minimum_required(VERSION 2.8)
project(ImageProcessing LANGUAGES C CXX CUDA)

#dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories( ./src)
include_directories( ./include)

add_executable(ImageProcessing ./src/image_processing.cu)

target_link_libraries(ImageProcessing ${OpenCV_LIBS})