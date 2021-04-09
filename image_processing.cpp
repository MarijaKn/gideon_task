#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;

int main()
{
    //std::string image_path = samples::findFile("starry_night.jpg");
    std::string image_path = "gideon_task/image_512x384.png";
    Mat img = imread(image_path, IMREAD_COLOR);

    // Check whether the image is imported correctly
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // Show image
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window

    if(k == 's')
    {
        imwrite("gideon_task/image_512x384.png", img);
    }
    return 0;
}