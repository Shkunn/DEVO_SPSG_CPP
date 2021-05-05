#include <array>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char * argv[])
{
    std::string image_path = samples::findFile("../data/img_1.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    resize(img,img,Size(img.cols/2, img.rows/2));
    std::cout << "image dimension (" << img.cols << "x" << img.rows << ")" << std::endl;

    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }

    return 0;
}
