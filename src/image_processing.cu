#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include <iostream>
#include <time.h>
#include <vector_types.h>

#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include "image_processing.h"
#include "image_processing.cuh"

#include "image_processing.cpp"
#include "image_processing_kernels.cu"

using namespace cv;
using namespace std;

int main() {

    // Load input file paths
    vector<string> image_paths;
    image_paths.push_back("./images/image_512x384.png");
    image_paths.push_back("./images/image_720x540.png");
    image_paths.push_back("./images/image_1024x768.png");

    // Iterate through the images, call the necessary kernels and perform measurements
    for (int i = 0; i < image_paths.size(); i++) {

        string image_path = image_paths.at(i);

        Mat img = imread(image_path);

        // Check whether the image is imported correctly
        if(img.empty())
        {
            cout << "Could not read the image: " << image_path << endl;
            return 1;
        }

        const unsigned int imgheight = img.rows;
        const unsigned int imgwidth = img.cols;

        cout << "Loaded " << imgwidth << " × " << imgheight << " image." << endl;

        // BGR to YUV444 conversion
        uchar3 *d_in;
        unsigned char *d_out_yuv444;

        cudaMalloc((void**)&d_in, imgheight*imgwidth*sizeof(uchar3));
        cudaMalloc((void**)&d_out_yuv444, imgheight*imgwidth*sizeof(unsigned char) * 3);

        cudaMemcpy(d_in, img.data, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyHostToDevice);

        double gpu_time = bgr_to_yuv444 (d_in, d_out_yuv444, imgheight, imgwidth);

        cout.precision(12);
        cout << "Execution time of bgr2yuv444 kernel is: "<< gpu_time << " sec." << endl;



        /// YUV444 to YUV422 conversion
        unsigned char *d_out_yuv422;
        cudaMalloc((void**)&d_out_yuv422, imgheight*imgwidth*sizeof(unsigned char) * 2);

        gpu_time = yuv444_to_yuv422 (d_out_yuv444, d_out_yuv422, imgheight, imgwidth);
        cout << "Execution time of yuv444toyuv422 kernel is: "<< gpu_time << " sec." << endl;

        // Conversion to BGR using cvtColor in order to display/save the image
        Mat yuv422_image(imgheight, imgwidth, CV_8UC2);
        Mat bgr_image(imgheight, imgwidth, CV_8UC3);

        cudaMemcpy(yuv422_image.data, d_out_yuv422, imgheight*imgwidth*sizeof(unsigned char) * 2, cudaMemcpyDeviceToHost);

        cvtColor(yuv422_image, bgr_image, CV_YUV2BGR_UYVY);

        // Save YUV422 image
        string filename = "bgr_to_yuv422_" + to_string(i+1) + ".png";
        imwrite(filename, bgr_image);



        /// YUV422 to single separate channels conversion
        unsigned char *d_y_channel;
        cudaMalloc((void**)&d_y_channel, imgheight*imgwidth*sizeof(unsigned char));

        unsigned char *d_u_channel;
        cudaMalloc((void**)&d_u_channel, imgheight*imgwidth*sizeof(unsigned char));

        unsigned char *d_v_channel;
        cudaMalloc((void**)&d_v_channel, imgheight*imgwidth*sizeof(unsigned char));

        gpu_time = extract_separate_channels(d_out_yuv422, d_y_channel, d_u_channel, d_v_channel, imgheight, imgwidth);
        cout << "Execution time of extract_channels kernel is: "<< gpu_time << " sec." << endl;

        // Save all 3 channels separately
        Mat output_y_channel(imgheight, imgwidth, CV_8UC1);
        cudaMemcpy(output_y_channel.data, d_y_channel, imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);

        filename = "y_channel_" + to_string(i+1) + ".png";
        imwrite(filename, output_y_channel);

        Mat output_uv_channels(imgheight, 0.5 * imgwidth, CV_8UC1);

        cudaMemcpy(output_uv_channels.data, d_u_channel, 0.5 * imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);

        filename = "u_channel_" + to_string(i+1) + ".png";
        imwrite(filename, output_uv_channels);

        cudaMemcpy(output_uv_channels.data, d_v_channel, 0.5 * imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);

        filename = "v_channel_" + to_string(i+1) + ".png";
        imwrite(filename, output_uv_channels);



        /// YUV422 to BGR conversion
        uchar3 *d_out_bgr;
        cudaMalloc((void**)&d_out_bgr, imgheight*imgwidth*sizeof(uchar3));

        gpu_time = yuv422_to_bgr (d_out_yuv422, d_out_bgr, imgheight, imgwidth);
        cout << "Execution time of yuv422tobgr kernel is: "<< gpu_time << " sec." << endl;

        Mat final_bgr_image(imgheight, imgwidth, CV_8UC3);
        cudaMemcpy(final_bgr_image.data, d_out_bgr, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyDeviceToHost);

        // Save final BGR image
        filename = "yuv442_to_bgr_" + to_string(i+1) + ".png";
        imwrite(filename, output_y_channel);

        cout << "-----------------------------------------------------------" << endl;

        cudaFree(d_in);
        cudaFree(d_out_yuv444);
        cudaFree(d_out_yuv422);
        cudaFree(d_y_channel);
        cudaFree(d_u_channel);
        cudaFree(d_v_channel);
        cudaFree(d_out_bgr);
    }

    return 0;
}