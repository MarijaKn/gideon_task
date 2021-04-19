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

#include "image_processing_kernels.cu"
#include "image_processing.cuh"

using namespace cv;
using namespace std;

int main() {

    string image_path = "./images/image_1024x768.png";

    Mat img = imread(image_path);

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

    // BGR to YUV444
    uchar3 *d_in;
    unsigned char *d_out_yuv444;

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

    /// YUV444 to YUV422
    unsigned char *d_out_yuv422;
    cudaMalloc((void**)&d_out_yuv422, imgheight*imgwidth*sizeof(unsigned char) * 2);

    unsigned int n_threads = 32;
    unsigned int n_blocks = (unsigned int) ceil((floor) (imgwidth * imgheight) / n_threads);

    start = clock();
    yuv444toyuv422<<<n_blocks, n_threads>>>(d_out_yuv444, d_out_yuv422, imgwidth*imgheight);
    cudaDeviceSynchronize();
    end = clock();

    gpu_time = (double) (end-start) / CLOCKS_PER_SEC;
    cout << "Execution time of yuv444toyuv422 kernel is:  "<< gpu_time << " sec." << endl;

    // COnversion to BGR using cvtColor in order to display/save the image
    Mat output_image(imgheight, imgwidth, CV_8UC2);
    Mat final_image(imgheight, imgwidth, CV_8UC3);

    cudaMemcpy(output_image.data, d_out_yuv422, imgheight*imgwidth*sizeof(unsigned char) * 2, cudaMemcpyDeviceToHost);

    cvtColor(output_image, final_image, CV_YUV2BGR_UYVY);

    // Show image
    imshow("Output YUV422 image", final_image);
    k = waitKey(0);

    /// YUV422 to single separate channels
    unsigned char *d_y_channel;
    cudaMalloc((void**)&d_y_channel, imgheight*imgwidth*sizeof(unsigned char));

    unsigned char *d_u_channel;
    cudaMalloc((void**)&d_u_channel, imgheight*imgwidth*sizeof(unsigned char));

    unsigned char *d_v_channel;
    cudaMalloc((void**)&d_v_channel, imgheight*imgwidth*sizeof(unsigned char));

    start = clock();
    extract_channels<<<n_blocks, n_threads>>>(d_out_yuv422, d_y_channel, d_u_channel, d_v_channel, imgwidth*imgheight);
    cudaDeviceSynchronize();
    end = clock();

    gpu_time = (double) (end-start) / CLOCKS_PER_SEC;
    cout << "Execution time of extract_channels kernel is:  "<< gpu_time << " sec." << endl;

    // Show/save all 3 channels separately
    Mat output_channels(imgheight, imgwidth, CV_8UC1);
    cudaMemcpy(output_channels.data, d_y_channel, imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Show image
    imshow("Y channel", output_channels);
    k = waitKey(0);

    Mat output_channels_1(imgheight, 0.5 * imgwidth, CV_8UC1);

    cudaMemcpy(output_channels_1.data, d_u_channel, 0.5 * imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    imshow("U channel", output_channels_1);
    k = waitKey(0);

    cudaMemcpy(output_channels_1.data, d_v_channel, 0.5 * imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    imshow("V channel", output_channels_1);
    k = waitKey(0);

    /// YUV422 to BGR
    uchar3 *d_out_bgr;
    cudaMalloc((void**)&d_out_bgr, imgheight*imgwidth*sizeof(uchar3));

    start = clock();
    yuv422tobgr<<<n_blocks, n_threads>>>(d_out_yuv422, d_out_bgr, imgheight*imgwidth);
    cudaDeviceSynchronize();
    end = clock();

    gpu_time = (double) (end-start) / CLOCKS_PER_SEC;
    cout << "Execution time of yuv422tobgr kernel is:  "<< gpu_time << " sec." << endl;

    Mat final_bgr_image(imgheight, imgwidth, CV_8UC3);
    cudaMemcpy(final_bgr_image.data, d_out_bgr, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyDeviceToHost);

    imshow("Final BGR image", final_bgr_image);
    k = waitKey(0);

    cudaFree(d_in);
    cudaFree(d_out_yuv444);
    cudaFree(d_out_yuv422);
    cudaFree(d_y_channel);
    cudaFree(d_u_channel);
    cudaFree(d_v_channel);
    cudaFree(d_out_bgr);

    return 0;
}