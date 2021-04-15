#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include <iostream>
#include <time.h>
#include <vector_types.h>

#include <stdio.h>

#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

__device__ unsigned char y_component (unsigned char b, unsigned char g, unsigned char r) {

    unsigned char y = 0.299 * r + 0.587 * g + 0.114 * b;

    return y;
}

__device__ unsigned char u_component (unsigned char y, unsigned char r, unsigned char delta) {

    unsigned char cr = (r - y) * 0.713 + delta;

    return cr;
}

__device__ unsigned char v_component (unsigned char y, unsigned char b, unsigned char delta) {

    unsigned char cb = (b - y) * 0.564 + delta;

    return cb;
}

__global__ void bgr2yuv444 (uchar3 *d_in, unsigned char *d_out, unsigned int imgwidth, unsigned int imgheight) {

    int column_idx = blockIdx.x*blockDim.x+threadIdx.x;
    int row_idx = blockIdx.y*blockDim.y+threadIdx.y;

    if ((row_idx > imgheight) || (column_idx > imgwidth)) {
        return;
    }

    int global_idx =  row_idx * imgwidth + column_idx;

    unsigned char b, g, r = 0;

    b = d_in[global_idx].x;
    g = d_in[global_idx].y;
    r = d_in[global_idx].z;

    unsigned char delta = 128;

    // In YUV444 every pixel has each of 3 components
    // Each pixel (global idx) reproduces 3 components on output and they are interspersed with each other

    unsigned char y = y_component(b, g, r);

    d_out[3 * global_idx]       = y;
    d_out[3 * global_idx + 1]   = u_component(y, r, delta);
    d_out[3 * global_idx + 2]   = v_component(y, b, delta);
}

__global__ void yuv444toyuv422 (unsigned char *d_in, unsigned char *d_out, unsigned int img_size) {

    int global_idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (global_idx > img_size) {
        return;
    }

    int y_in_idx  = 3 * global_idx;
    int u_in_idx  = y_in_idx + 1;
    int v_in_idx  = y_in_idx + 2;

    int y_out_idx = 2 * global_idx;
    int u_out_idx = 2 * y_out_idx + 1;
    int v_out_idx = 2 * y_out_idx + 3;

    d_out[y_out_idx] = d_in[y_in_idx];
    d_out[u_out_idx] = d_in[u_in_idx];
    d_out[v_out_idx] = d_in[v_in_idx];

    // if (global_idx < 20) {
    //     printf("global_idx = %d | y_in_idx = %d, u_in_idx = %d, v_in_idx = %d | y_out_idx = %d, u_out_idx = %d, v_out_idx = %d\n",
    //             global_idx, y_in_idx, u_in_idx, v_in_idx, y_out_idx, u_out_idx, v_out_idx);
    // }

}

int main()
{
    string image_path = "gideon_task/image_512x384.png";
    Mat img = imread(image_path, IMREAD_COLOR);

    // Check whether the image is imported correctly
    if(img.empty())
    {
        cout << "Could not read the image: " << image_path << endl;
        return 1;
    }

    // Show image
    imshow("Loaded BGR image", img);
    int k = waitKey(0);

    const unsigned int imgheight = img.rows;
    const unsigned int imgwidth = img.cols;

    cout << "Image height  =  " << imgheight << endl;
    cout << "Image width   =  "  << imgwidth << endl;

    // Define input and output
    uchar3 *d_in;
    unsigned char *d_out_yuv444;

    // Allocate device memory for input and output
    cudaMalloc((void**)&d_in, imgheight*imgwidth*sizeof(uchar3));
    cudaMalloc((void**)&d_out_yuv444, imgheight*imgwidth*sizeof(unsigned char) * 3);

    cudaMemcpy(d_in, img.data, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    clock_t start, end;
    start = clock();

    bgr2yuv444<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out_yuv444, imgwidth, imgheight);
    cudaDeviceSynchronize();

    end = clock();

    double gpu_time = (double) (end-start) / CLOCKS_PER_SEC;
    cout.precision(12);

    cout << "Execution time of bgr2yuv444 kernel is:  "<< gpu_time << " sec." << endl;

    unsigned char *d_out_yuv422;
    cudaMalloc((void**)&d_out_yuv422, imgheight*imgwidth*sizeof(unsigned char) * 2);

    unsigned int n_threads = 32;
    unsigned int n_blocks = (unsigned int) ceil((floor) (imgwidth * imgheight) / n_threads);

    cout << "n_threads = " << n_threads  << endl;
    cout << "n_blocks = " << n_blocks << endl;

    start = clock();

    yuv444toyuv422<<<n_blocks, n_threads>>>(d_out_yuv444, d_out_yuv422, imgwidth*imgheight);

    end = clock();

    gpu_time = (double) (end-start) / CLOCKS_PER_SEC;
    cout.precision(12);

    cout << "Execution time of yuv444toyuv422 kernel is:  "<< gpu_time << " sec." << endl;

    Mat output_image = Mat(imgheight, imgwidth, CV_8UC2, Scalar(255));
    Mat final_image = Mat(imgheight, imgwidth, CV_8UC3, Scalar(255));

    cudaMemcpy(output_image.data, d_out_yuv422, imgheight*imgwidth*sizeof(unsigned char) * 2, cudaMemcpyDeviceToHost);

    cvtColor(output_image, final_image, COLOR_YUV2BGR_UYVY, 3);

    // Show image
    imshow("Output YUV422 image", final_image);
    k = waitKey(0);

    cudaFree(d_in);
    cudaFree(d_out_yuv444);
    cudaFree(d_out_yuv422);


    return 0;
}