//MosaicGenerator.cpp - inputs image and outputs an image with a mosaic tranformation. 
// Aishwarya Pani and Tabitha Roemish

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
Mat calcLuminance(Mat image);
Mat calcGradient(Mat image);
Mat calcGVFMap(Mat image);
Mat calcNonMax(Mat image);
Mat calcTileAngles(Mat image);
Mat placeTiles(Mat image);


//main
//preconditions:
//postconditions:
int main(int argc, char* argv[]) {

    // Step 1: Read an input image from directory
    Mat image = imread("sqrl.jpg");

    // Step 2: Calculate the luminance of the image
    Mat luminance = calcLuminance(image);

    // Step 3: Calculate Robert's gradient of the luminance image (add tile size?)
    Mat robert = calcGradient(luminance);

    // Step 4: Calculate Gradient Vector Flow Map (add tile size?)
    Mat gvf = calcGVFMap(robert);

    // Step 5: Calculate NonMaximumSuppression
    Mat nonMax = calcNonMax(gvf);

    // Step 6: Calculate tile Angles  (add tile size ? )
    Mat angleMap = calcTileAngles(nonMax);

    // Step 7: Place tiles  (add tile size ? )
    Mat mosaic = placeTiles(angleMap);

    // Step 8: Display Images
    namedWindow("Input Image", WINDOW_NORMAL);
    imshow("Input Image", image);
    namedWindow("Luminance", WINDOW_NORMAL);
    imshow("Luminance", luminance);
    namedWindow("Gradient", WINDOW_NORMAL);
    imshow("Gradient", robert);
    //namedWindow("GVF", WINDOW_NORMAL);
    //imshow("GVF", gvf);
    //namedWindow("nonMax", WINDOW_NORMAL);
    //imshow("nonMax", nonMax);
    //namedWindow("angleMap", WINDOW_NORMAL);
    //imshow("angleMap", angleMap);
    //namedWindow("mosaic", WINDOW_NORMAL);
    //imshow("mosaic", mosaic);
    waitKey(0);
    return 0;
}

// Function to calculate the luminance of an image
Mat calcLuminance(Mat image)
{
    Mat luminance;
    cvtColor(image, luminance, COLOR_BGR2GRAY);
    return luminance;
}

// Function to calculate Robert's gradient of an image
/* Steps -
   1. The Roberts kernels are defined as 2x2 matrices.
   2. Loops through each pixel in the input image.
   3. The gradient is calculated using the Roberts kernels.
   4. Then the magnitude of the gradient is calculated.
   5. Sets the output pixel to the magnitude of the gradient. */
Mat calcGradient(Mat image)
{
    int kernel_x[2][2] = { {1, 0}, {0, -1} };
    int kernel_y[2][2] = { {0, 1}, {-1, 0} };

    Mat robert_image = Mat::zeros(image.size(), CV_8UC1);

    for (int i = 0; i < image.rows - 1; i++)
    {
        for (int j = 0; j < image.cols - 1; j++)
        {
            int gradient_x = 0;
            int gradient_y = 0;

            for (int k = 0; k < 2; k++)
            {
                for (int l = 0; l < 2; l++)
                {
                    gradient_x += image.at<uchar>(i + k, j + l) * kernel_x[k][l];
                    gradient_y += image.at<uchar>(i + k, j + l) * kernel_y[k][l];
                }
            }

            int magnitude = sqrt(pow(gradient_x, 2) + pow(gradient_y, 2));

            robert_image.at<uchar>(i, j) = magnitude;
        }
    }
    return robert_image;
}

//calcGVFMap -  calculates the gradient vector flow for each tile
//preconditions:
//postconditions:
Mat calcGVFMap(Mat image) {}

//calcNonMax - calculates the non-maximum suppression for each tile
//preconditions:
//postconditions:
Mat calcNonMax(Mat image) {}

//calcTileAngles - calculate the angle for each tile
//preconditions:
//postconditions:
Mat calcTileAngles(Mat image) {}

//placeTiles - this final method creates the final mosaic image by placing the tiles according to the angleMap
//preconditions:
//postconditions:
Mat placeTiles(Mat image) {}