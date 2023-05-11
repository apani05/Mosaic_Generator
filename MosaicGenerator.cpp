//MosaicGenerator.cpp - inputs image and outputs an image with a mosaic tranformation. 
// Aishwarya Pani and Tabitha Roemish

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

typedef double PGdouble;

Mat calcLuminance(Mat image);
Mat calcGradient(Mat image);
Mat calcGVFField(Mat image, vector<Point>* v);
Mat calcNonMax(Mat image);
Mat placeTiles(Mat image, Mat nonMax, vector<Point>* v, int tsize);


//main
//preconditions:
//postconditions:
int main(int argc, char* argv[]) {

    vector<Point> v;
    int tilesize = 1; // 1 pixel X 1 pixel

    // Step 1: Read an input image from directory
    Mat image = imread("lena.jpg");

    // Step 2: Calculate the luminance of the image
    Mat luminance = calcLuminance(image);
    
    // Step 3: Calculate Robert's gradient of the luminance image (add tile size?)
    Mat robert = calcGradient(luminance);

    // Step 4 - 5: Calculate Gradient Vector Flow Map (add tile size?)
    Mat gvf = calcGVFField(robert, &v);

    // Step 6: Calculate NonMaximumSuppression
    Mat nonMax = calcNonMax(gvf);

    // Step 7-23: Calculate tile Angles and place tiles  (add tile size ? )
    Mat mosaic = placeTiles(image, nonMax, &v, tilesize);


    // Display Images
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
Mat calcGVFField(Mat image, vector<Point>* v) {
    Mat gvf = Mat::zeros(image.size(), CV_8UC1);
    //input roberts image
    //loop over image and calculate the gradient field vector
    //calculate the magnitude (u^2 + v^2)^1/2
    //place that magnitude in the output image
    return gvf;
}

//calcNonMax - calculates the non-maximum suppression for each tile
//preconditions: n/a
//postconditions: outputs a mat image that has nonmaxsuppression edges detected
Mat calcNonMax(Mat image) {
    Mat nonMax = Mat::zeros(image.size(), CV_8UC1);
    Mat mag, angle = Mat::zeros(image.size(), CV_32F);
    const int direction = 22.5;
    float p1, p2;
    Mat input;
    image.convertTo(input, CV_32F);
    cartToPolar(input, input, mag, angle, true);
    for (int i = 1; i < input.rows - 1; i++)
    {
        for (int j = 1; j < input.cols - 1; j++)
        {
            if ((angle.at<float>(i, j) >= 0 && angle.at<float>(i, j) < direction) ||
                (angle.at<float>(i, j) <= direction * 7 && angle.at<float>(i, j) <= direction * 8) ||
                (angle.at<float>(i, j) < 0 && angle.at<float>(i, j) >= direction * -1) ||
                (angle.at<float>(i, j) >= direction * -8 && angle.at<float>(i, j) < direction * -7))
            {//horizontal
                p1 = mag.at<float>(i, j + 1);
                p2 = mag.at<float>(i, j - 1);
            }
            else if ((angle.at<float>(i, j) >= direction && angle.at<float>(i, j) < direction * 3) ||
                (angle.at<float>(i, j) >= direction * -7 && angle.at<float>(i, j) < direction * -5))
            {//diagonal 45
                p1 = mag.at<float>(i + 1, j + 1);
                p2 = mag.at<float>(i - 1, j - 1);
            }
            else if ((angle.at<float>(i, j) >= direction * 3 && angle.at<float>(i, j) < direction * 5) ||
                (angle.at<float>(i, j) >= direction * -5 && angle.at<float>(i, j) < direction * -3))
            {//vertical
                p1 = mag.at<float>(i + 1, j);
                p2 = mag.at<float>(i - 1, j);
            }
            else //diagonal 135
            {
                p1 = mag.at<float>(i + 1, j - 1);
                p2 = mag.at<float>(i - 1, j + 1);
            }
            if (mag.at<float>(i, j) >= p1 && mag.at<float>(i, j) >= p2)
                nonMax.at<uchar>(i, j) = mag.at<float>(i, j);
            else
                nonMax.at<uchar>(i, j) = 0;
        }
    }

    return nonMax;
}

PGdouble** pgDmatrix(int nrl, int nrh, int ncl, int nch)
{
    int j;
    long bufsize, bufptr;
    PGdouble** m;

    bufsize = (nrh - nrl + 1) * sizeof(PGdouble*) + (nrh - nrl + 1) * (nch - ncl + 1) * sizeof(PGdouble);

    m = (PGdouble**)malloc(bufsize);
    m -= nrl;

    bufptr = ((long)(m + nrl)) + (nrh - nrl + 1) * sizeof(PGdouble*);
    for (j = nrl; j <= nrh; j++)
    {
        m[j] = ((PGdouble*)(bufptr + (j - nrl) * (nch - ncl + 1) * sizeof(PGdouble)));
        m[j] -= ncl;
    }

    return m;
}

void pgFreeDmatrix(PGdouble** m, int nrl, int nrh, int ncl, int nch)
{
    free((char*)(m + nrl));
}

void GVFC(int YN, int XN, double* f, double* ou, double* ov, double mu, int ITER)
{
    double mag2, temp, tempx, tempy, fmax, fmin;
    int count, x, y, XN_1, XN_2, YN_1, YN_2;

    PGdouble** fx, ** fy, ** u, ** v, ** Lu, ** Lv, ** g, ** c1, ** c2, ** b;

    // define constants and create row-major double arrays
    XN_1 = XN - 1;
    XN_2 = XN - 2;
    YN_1 = YN - 1;
    YN_2 = YN - 2;
    fx = pgDmatrix(0, YN_1, 0, XN_1);
    fy = pgDmatrix(0, YN_1, 0, XN_1);
    u = pgDmatrix(0, YN_1, 0, XN_1);
    v = pgDmatrix(0, YN_1, 0, XN_1);
    Lu = pgDmatrix(0, YN_1, 0, XN_1);
    Lv = pgDmatrix(0, YN_1, 0, XN_1);
    g = pgDmatrix(0, YN_1, 0, XN_1);
    c1 = pgDmatrix(0, YN_1, 0, XN_1);
    c2 = pgDmatrix(0, YN_1, 0, XN_1);
    b = pgDmatrix(0, YN_1, 0, XN_1);

    // Normalize the edge map to [0,1]
    fmax = -1e10;
    fmin = 1e10;
    for (x = 0; x <= YN * XN - 1; x++) {
        fmax = std::max(fmax, f[x]);
        fmin = std::min(fmin, f[x]);
    }

    if (fmax == fmin)
        std::cout << "Edge map is a constant image." << std::endl;

    for (x = 0; x <= YN * XN - 1; x++)
        f[x] = (f[x] - fmin) / (fmax - fmin);

        /**************** II: Compute edge map gradient *****************/
    /* I.1: Neumann boundary condition:
     *      zero normal derivative at boundary
     */
     /* Deal with corners */
    fx[0][0] = fy[0][0] = fx[0][XN_1] = fy[0][XN_1] = 0;
    fx[YN_1][XN_1] = fy[YN_1][XN_1] = fx[YN_1][0] = fy[YN_1][0] = 0;

    /* Deal with left and right column */
    for (y = 1; y <= YN_2; y++) {
        fx[y][0] = fx[y][XN_1] = 0;
        fy[y][0] = 0.5 * (f[y + 1] - f[y - 1]);
        fy[y][XN_1] = 0.5 * (f[y + 1 + XN_1 * YN] - f[y - 1 + XN_1 * YN]);
    }

    /* Deal with top and bottom row */
    for (x = 1; x <= XN_2; x++) {
        fy[0][x] = fy[YN_1][x] = 0;
        fx[0][x] = 0.5 * (f[(x + 1) * YN] - f[(x - 1) * YN]);
        fx[YN_1][x] = 0.5 * (f[YN_1 + (x + 1) * YN] - f[YN_1 + (x - 1) * YN]);
    }

    /* I.2: Compute interior derivative using central difference */
    for (y = 1; y <= YN_2; y++)
        for (x = 1; x <= XN_2; x++) {
            /* NOTE: f is stored in column major */
            fx[y][x] = 0.5 * (f[y + (x + 1) * YN] - f[y + (x - 1) * YN]);
            fy[y][x] = 0.5 * (f[y + 1 + x * YN] - f[y - 1 + x * YN]);
        }

    /******* III: Compute parameters and initializing arrays **********/
    temp = -1.0 / (mu * mu);
    for (y = 0; y <= YN_1; y++)
        for (x = 0; x <= XN_1; x++) {
            tempx = fx[y][x];
            tempy = fy[y][x];
            /* initial GVF vector */
            u[y][x] = tempx;
            v[y][x] = tempy;
            /* gradient magnitude square */
            mag2 = tempx * tempx + tempy * tempy;

            g[y][x] = mu;
            b[y][x] = mag2;

            c1[y][x] = b[y][x] * tempx;
            c2[y][x] = b[y][x] * tempy;
        }

    /* free memory of fx and fy */
    pgFreeDmatrix(fx, 0, YN_1, 0, XN_1);
    pgFreeDmatrix(fy, 0, YN_1, 0, XN_1);

    /************* Solve GVF = (u,v) iteratively ***************/
    for (count = 1; count <= ITER; count++) {
        /* IV: Compute Laplace operator using Neuman condition */
        /* IV.1: Deal with corners */
        Lu[0][0] = (u[0][1] + u[1][0]) * 0.5 - u[0][0];
        Lv[0][0] = (v[0][1] + v[1][0]) * 0.5 - v[0][0];
        Lu[0][XN_1] = (u[0][XN_2] + u[1][XN_1]) * 0.5 - u[0][XN_1];
        Lv[0][XN_1] = (v[0][XN_2] + v[1][XN_1]) * 0.5 - v[0][XN_1];
        Lu[YN_1][0] = (u[YN_1][1] + u[YN_2][0]) * 0.5 - u[YN_1][0];
        Lv[YN_1][0] = (v[YN_1][1] + v[YN_2][0]) * 0.5 - v[YN_1][0];
        Lu[YN_1][XN_1] = (u[YN_1][XN_2] + u[YN_2][XN_1]) * 0.5 - u[YN_1][XN_1];
        Lv[YN_1][XN_1] = (v[YN_1][XN_2] + v[YN_2][XN_1]) * 0.5 - v[YN_1][XN_1];

        /* IV.2: Deal with left and right columns */
        for (y = 1; y <= YN_2; y++) {
            Lu[y][0] = (2 * u[y][1] + u[y - 1][0] + u[y + 1][0]) * 0.25 - u[y][0];
            Lv[y][0] = (2 * v[y][1] + v[y - 1][0] + v[y + 1][0]) * 0.25 - v[y][0];
            Lu[y][XN_1] = (2 * u[y][XN_2] + u[y - 1][XN_1] + u[y + 1][XN_1]) * 0.25 - u[y][XN_1];
            Lv[y][XN_1] = (2 * v[y][XN_2] + v[y - 1][XN_1] + v[y + 1][XN_1]) * 0.25 - v[y][XN_1];
        }
        /* IV.3: Deal with top and bottom rows */
        for (x = 1; x <= XN_2; x++) {
            Lu[0][x] = (2 * u[1][x] + u[0][x - 1] + u[0][x + 1]) * 0.25 - u[0][x];
            Lv[0][x] = (2 * v[1][x] + v[0][x - 1] + v[0][x + 1]) * 0.25 - v[0][x];
            Lu[YN_1][x] = (2 * u[YN_2][x] + u[YN_1][x - 1] + u[YN_1][x + 1]) * 0.25 - u[YN_1][x];
            Lv[YN_1][x] = (2 * v[YN_2][x] + v[YN_1][x - 1] + v[YN_1][x + 1]) * 0.25 - v[YN_1][x];
        }

        /* IV.4: Compute interior */
        for (y = 1; y <= YN_2; y++) {
            for (x = 1; x <= XN_2; x++) {
                Lu[y][x] = (u[y][x - 1] + u[y][x + 1] + u[y - 1][x] + u[y + 1][x]) * 0.25 - u[y][x];
                Lv[y][x] = (v[y][x - 1] + v[y][x + 1] + v[y - 1][x] + v[y + 1][x]) * 0.25 - v[y][x];
            }
        }

        /******** V: Update GVF ************/
        for (y = 0; y <= YN_1; y++) {
            for (x = 0; x <= XN_1; x++) {
                temp = u[y][x];
                    u[y][x] += Lu[y][x];
                    Lu[y][x] = temp;

                    temp = v[y][x];
                    v[y][x] += Lv[y][x];
                    Lv[y][x] = temp;
            }
        }
    }

    /********** VI: Normalize the GVF ***********/
    for (y = 0; y <= YN_1; y++) {
        for (x = 0; x <= XN_1; x++) {
            ou[x * YN + y] = u[y][x];
            ov[x * YN + y] = v[y][x];
        }
    }

    /********** VII: Free memory ***********/
    pgFreeDmatrix(u, 0, YN_1, 0, XN_1);
    pgFreeDmatrix(v, 0, YN_1, 0, XN_1);
    pgFreeDmatrix(Lu, 0, YN_1, 0, XN_1);
    pgFreeDmatrix(Lv, 0, YN_1, 0, XN_1);
    pgFreeDmatrix(g, 0, YN_1, 0, XN_1);
    pgFreeDmatrix(c1, 0, YN_1, 0, XN_1);
    pgFreeDmatrix(c2, 0, YN_1, 0, XN_1);
    pgFreeDmatrix(b, 0, YN_1, 0, XN_1);

}

//placeTiles - this final method creates the final mosaic image by placing the tiles according to the angleMap
//preconditions:
//postconditions:
Mat placeTiles(Mat image, Mat nonMax, vector<Point>* v, int tileSize) {
    Mat mosaic = Mat::zeros(image.size(), CV_8UC3);
    struct pixel
    {
        int x;
        int y;
        bool mark;
    };
    queue<pixel> Q;
    float angleAlpha;
    float angleBeta;
    //add points to queue if above threshold
    //sort queue desc
    while (!Q.empty())
    {
        pixel p = Q.front();
        if (p.mark == false) {
            //angleAlpha = atan(u(i,j)/v(i,j)
            //if(!checkOverlap(image, row, col, angle)
                //place tile at angle, maybe use rotaterect?
        }

        Q.pop();
    }
    return mosaic;
}

//checkOverlap...not sure what this needs yet
bool checkOverlap(Mat image, int pixRow, int pixCol, float angle) {
    //return true if overlap
    bool overlap = false;

    return overlap;
}