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

// u_component
__device__ unsigned char cr_component (unsigned char y, unsigned char r, unsigned char delta) {

    unsigned char cr = (r - y) * 0.713 + delta;

    return cr;
}

// y component
__device__ unsigned char cb_component (unsigned char y, unsigned char b, unsigned char delta) {

    unsigned char cb = (b - y) * 0.564 + delta;

    return cb;
}

__global__ void bgr2ycrcb (uchar3 *d_in, unsigned char *d_out, unsigned int imgwidth, unsigned int imgheight) {

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

    // In YUV444 every thread reproduces 3 otuputs, y u and v component
    // for (int i = 0; i < 3; i++) {

    //     unsigned char value = 0;
    //     unsigned char y = y_component(b, g, r);

    //     if (i == 0) {
    //         value = y;
    //     } else if (i == 1) {
    //         value = cr_component(y, r, delta);
    //     } else {
    //         value = cb_component(y, b, delta);
    //     }

    //     d_out[3 * global_idx + i] = value;
    // }

    for (int i = 0; i < 2; i++) {

        unsigned char value = 0;
        unsigned char y = y_component(b, g, r);

        if (i == 1) {
            value = y;
        } else {

            if (global_idx % 2 == 0) {
                value = cr_component(y, r, delta);
            } else if (global_idx % 2 == 1) {
                value = cb_component(y, b, delta);
            }
        }

        d_out[2 * global_idx + i] = value;
    }
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

    // This means object Mat was created, a simple image containter
    // Image dimensions imgheight x imagewidth
    // CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels
    // The cv::Scalar is four element short vector. Specify it and you can initialize all matrix points with a custom value
    Mat output_image = Mat(imgheight, imgwidth, CV_8UC3, Scalar::all(0));

    // Define input and output
    uchar3 *d_in;
    unsigned char *d_out;

    // Allocate device memory for input and output
    cudaMalloc((void**)&d_in, imgheight*imgwidth*sizeof(uchar3));
    cudaMalloc((void**)&d_out, imgheight*imgwidth*sizeof(unsigned char) * 2);

    cudaMemcpy(d_in, img.data, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    clock_t start, end;
    start = clock();

    bgr2ycrcb<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, imgwidth, imgheight);
    cudaDeviceSynchronize();

    end = clock();

    double gpu_time = (double) (end-start) / CLOCKS_PER_SEC;
    cout.precision(12);

    cout << "Execution time of bgr2ycrcb kernel is:  "<< gpu_time << " sec." << endl;

    cudaMemcpy(output_image.data, d_out, imgheight*imgwidth*sizeof(unsigned char) * 2, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    // Show output image
    imshow("YUV420 image", output_image);
    k = waitKey(0);

    return 0;
}