// MosaicGenerator.cpp 
/* This program takes an input image and outputs an image with a digital mosaic tranformation
   using methods from the OpenCV library.
 * Images are displayed on the screen in the given order - Input Image, Luminance, Robert Gradient,
   GVF, Non-Max Suppression, Mosaic Image. */
// Authors: Aishwarya Pani and Tabitha Roemish
// Completed on: June 6, 2023

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <iostream>

enum { red, green, blue };
using namespace cv;
using namespace std;

// Function declarations - The 5 main methods to achieve a Mosaic are each converted to a function.
Mat calcLuminance(Mat image);
Mat calcGradient(Mat image);
Mat calcGVFField(Mat image, Mat gx, Mat gy);
Mat calcNonMax(Mat image);
Mat placeTiles(Mat image, Mat nonMax, Mat gx, Mat gy, int tileSize);

// Extra Helper methods
void setTile(RotatedRect rRect, Mat image, Mat mosaic);
RotatedRect positionTile(RotatedRect rRect, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc);
RotatedRect shift(int quadrant, int startingPoint, RotatedRect rRect, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc);
bool checkPlacementOk(RotatedRect tile, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc);
Point getPerpendicularTile(RotatedRect rRect, int direction, Size s, float angleAlpha);
void displayImage(string name, Mat image);
float getAvgVal(Mat tile);
void getVertices(Point* arr, RotatedRect rRect);


/* main - The main function reads an input image from the directory and generates a mosaic image digitally.
 1.Preconditions: lena.jpg exists in the code directory and is a valid JPG.
 2.Postconditions: Images are displayed on the screen in the given order - Input Image, Luminance,
   Robert Gradient, GVF, Non-Max Suppression, Mosaic Image.
   The main method saves the final output image in the directory. */

int main(int argc, char* argv[]) {

    // Step 1: Read an input image from the directory
    Mat image = imread("lena.jpg"); // Change the image filename as needed
    resize(image, image, Size(667, 667));

    // Step 2: Generate the luminance image
    Mat luminance = calcLuminance(image);

    // Step 3: Calculate Robert's gradient of the luminance image 
    Mat robert = calcGradient(luminance);

    // Step 4 - 5: Calculate Gradient Vector Flow Map
    Mat gx = Mat::zeros(image.size(), CV_8UC1);
    Mat gy = Mat::zeros(image.size(), CV_8UC1);
    Mat gvf = calcGVFField(robert, gx, gy);

    // Step 6: Calculate Non-Maximum Suppression
    Mat nonMax = calcNonMax(gvf);

    // Step 7-23: Calculate tile Angles and place tiles accordingly 
    int const tilesize = 4; // 1 pixel X 1 pixel
    Mat mosaic = placeTiles(image, nonMax, gx, gy, tilesize);

    // Display Images
    displayImage("Input Image", image);
    displayImage("Luminance", luminance);
    displayImage("Robert Gradient", robert);
    displayImage("GVF", gvf);
    displayImage("Non-Max Suppression", nonMax);
    displayImage("Mosaic Image", mosaic);
    waitKey(0);

    // Save the output image in the directory
    imwrite("MosaicFinalImage.jpg", mosaic);
    return 0;
}

/* displayImage - This function takes an image and displays it on the screen.
 1.Preconditions: A valid image and its name are provided as input.
 2.Postconditions: Displays the image on the screen. */

void displayImage(string name, Mat image)
{
    namedWindow(name, WINDOW_NORMAL);
    resizeWindow(name, image.cols, image.rows);
    imshow(name, image);
}

/* calcLuminance - This function calculates the luminance of an image and displays a grayscale image.
 1.Preconditions: A valid image is provided as input.
 2.Postconditions : Returns a grayscale image. */

Mat calcLuminance(Mat image)
{
    Mat luminance;
    cvtColor(image, luminance, COLOR_BGR2GRAY);
    return luminance;
}

// calcGradient - This function calculates Robert's gradient of a grayscale image
/* Steps -
   1. The Roberts kernels are defined as 2x2 matrices.
   2. Loops through each pixel in the input image.
   3. The gradient is calculated using the Roberts kernels.
   4. Then the magnitude of the gradient is calculated.
   5. Sets the output pixel to the magnitude of the gradient.
* Preconditions: A valid grayscale image is provided as input.
* Postconditions : Returns a Robert's gradient image. */

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

// calcGVFField - This function calculates the change in the x and y coordinates for each pixel
/* Steps:
   1. Calculate the gradient in x and y directions using Scharr function
   2. Return the calculated GVF image which contains the gradient change in x and y combined*/
// Preconditions: The input image, gx, and gy are valid and initialized Mat objects.
// Postconditions: Returns a Mat object representing the gradient changes in x and y coordinates.

Mat calcGVFField(Mat image, Mat gx, Mat gy)
{
    int depth = CV_8UC1;
    int scale = 1;
    int delta = 0;

    Mat gvf = Mat::zeros(image.size(), CV_8UC1);
    Scharr(image, gx, depth, 1, 0, scale);
    Scharr(image, gy, depth, 0, 1, scale);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            Scalar x_val = gx.at<uchar>(i, j);
            Scalar y_val = gy.at<uchar>(i, j);
            gvf.at<uchar>(i, j) = (x_val.val[0], y_val.val[0]);
        }
    }
    return gvf;
}

// calcNonMax - This function calculates the non-maximum suppression for each tile
// Steps:
//   1. Compute the magnitude and angle of gradients using cartToPolar function
//   2. Perform non-maximum suppression by comparing neighboring pixel values
//   3. Store the result in a new Mat object and return it

// Preconditions: Takes a valid input image.
// Postconditions: Returns a Mat object representing the non-maximum suppression result.

Mat calcNonMax(Mat image)
{
    Mat nonMax = Mat::zeros(image.size(), CV_8UC1);
    Mat mag, angle = Mat::zeros(image.size(), CV_32F);
    Mat input;

    const int direction = 22.5;
    float p1, p2;
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
            { //Horizontal direction
                p1 = mag.at<float>(i, j + 1);
                p2 = mag.at<float>(i, j - 1);
            }
            else if ((angle.at<float>(i, j) >= direction && angle.at<float>(i, j) < direction * 3) ||
                (angle.at<float>(i, j) >= direction * -7 && angle.at<float>(i, j) < direction * -5))
            { //Diagonal 45-degree direction
                p1 = mag.at<float>(i + 1, j + 1);
                p2 = mag.at<float>(i - 1, j - 1);
            }
            else if ((angle.at<float>(i, j) >= direction * 3 && angle.at<float>(i, j) < direction * 5) ||
                (angle.at<float>(i, j) >= direction * -5 && angle.at<float>(i, j) < direction * -3))
            { //Vertical direction
                p1 = mag.at<float>(i + 1, j);
                p2 = mag.at<float>(i - 1, j);
            }
            else
            { //Diagonal 135-degree direction
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

// placeTiles - This final method creates the final mosaic image by placing the tiles according to the angleMap
// Preconditions:
//   - The input image is a valid and initialized Mat object.
//   - The nonMax, gx, and gy Mats are valid and initialized Mat objects.
//   - The tileSize specifies the desired size of each tile in the mosaic.
// Postconditions:
//   - Returns a Mat object representing the final mosaic image.

Mat placeTiles(Mat image, Mat nonMax, Mat gx, Mat gy, int tileSize)
{
    float avgLum = getAvgVal(nonMax);
    int threshold_h = avgLum * .80; //Set the thresholds for determining high and low gradient tiles
    int threshold_l = avgLum * .20;
    Mat mosaic(Size(image.cols, image.rows), image.type(), cv::Scalar(102.0, 102.0, 102.0)); //dark slate grey mosaic image
    queue<Point> Q;
    float angleAlpha = 0;
    float angleBeta = 0;
    float angleGamma = 0;

    //Set the desired mosaic size and number of tiles in the x and y directions
    Size mosaicSize(tileSize, tileSize);
    float tileR = tileSize / 2;
    int numTilesX = image.cols / mosaicSize.width;
    int numTilesY = image.rows / mosaicSize.height;

    //Divide gx and gy into tiles
    Mat gxTiles = Mat::zeros(Size(numTilesX, numTilesY), CV_8UC1);
    Mat gyTiles = Mat::zeros(Size(numTilesX, numTilesY), CV_8UC1);

    //Set up a map of RotatedRects to check for tile overlap
    RotatedRect emptyCell(Point2f(-1, -1), mosaicSize, 0);
    std::vector<std::vector<RotatedRect> > tileMap(numTilesX, std::vector<RotatedRect>(numTilesY, emptyCell));
    Size mapSize(tileMap.size(), tileMap[0].size());

    //Fill the queue and sort it based on gradient values
    vector<tuple<float, Point>> v;
    int it = 0;
    for (int i = 0; i < numTilesX; i++)
    {
        for (int j = 0; j < numTilesY; j++)
        {
            Rect tileRect(i * mosaicSize.width, j * mosaicSize.height, mosaicSize.width, mosaicSize.height);
            Mat tile = nonMax(tileRect);
            Mat gxTile = gx(tileRect);
            Mat gyTile = gy(tileRect);

            avgLum = getAvgVal(tile);
            if (avgLum > threshold_h)
            {
                v.push_back(make_tuple(avgLum, Point(i, j)));
            }

            //Fill gxTiles and gyTiles with average change
            float avgXChange = getAvgVal(gxTile);
            float avgYChange = getAvgVal(gyTile);
            gxTiles.at<uchar>(i, j) = avgXChange;
            gyTiles.at<uchar>(i, j) = avgYChange;
        }
    }

    //Sort vector from largest gradient to smallest
    sort(v.begin(), v.end(),
        [](const tuple<float, Point>& a,
            const tuple<float, Point>& b) -> bool
        {
            return std::get<0>(a) > std::get<0>(b);
        });

    //Add the sorted tiles to the queue
    for (int i = 0; i < v.size(); i++)
    {
        Q.push(get<1>(v[i]));
    }

    //Start placing edge and adjacent tiles
    while (!Q.empty())
    {
        Point temp = Q.front();
        Q.pop();
        if (tileMap[temp.x][temp.y].center.x == -1 && tileMap[temp.x][temp.y].center.y == -1)
        {
            //Set tiles > threshold_h 
            Point recCenter(temp.x * mosaicSize.width + tileR, temp.y * mosaicSize.height + tileR);
            angleAlpha = atan2(gyTiles.at<uchar>(temp.x, temp.y), gxTiles.at<uchar>(temp.x, temp.y)) * (180 / CV_PI);
            RotatedRect rRect(recCenter, mosaicSize, angleAlpha); //initial tile
            RotatedRect rSet = positionTile(rRect, tileMap, mapSize, Point(temp.x, temp.y)); //shifted tile

            if (rSet.center.x != -1 && rSet.center.y != -1)
            {
                setTile(rSet, image, mosaic);
                tileMap[temp.x][temp.y] = rSet; // add tile to map
            }

            //Set tiles >  threshold_l 
            int direction = 90;
            RotatedRect newrRec = rRect;
            while (true)
            {
                //Start looking for adjacent tiles perpendicular to edge tile
                Point p = getPerpendicularTile(newrRec, direction, image.size(), angleAlpha);
                if (p == Point(-1, -1))
                {
                    if (direction == -90)
                        break;
                    newrRec = rRect;
                    direction = -90;
                    p = getPerpendicularTile(newrRec, direction, image.size(), angleAlpha);
                    if (p == Point(-1, -1))
                        break; //end search
                }

                Point tl(p.x - rRect.size.height / 2.0, rRect.center.y - rRect.size.width / 2.0);
                Rect rec(tl, rRect.size);
                Mat tile = nonMax(rec);
                avgLum = getAvgVal(tile);
                Point tileMapLoc((p.x - tileR) / mosaicSize.width, (p.y - tileR) / mosaicSize.height);
                if (avgLum > threshold_l || (tileMap[tileMapLoc.x][tileMapLoc.y].center.x != -1
                    && tileMap[tileMapLoc.x][tileMapLoc.y].center.y != -1))
                {
                    //Tile has gradient less than threshold_l or tile is already set
                    if (direction == -90)
                    {
                        break;
                    }
                    direction = -90;
                    newrRec = rRect;
                }
                else
                {
                    angleBeta = atan2(gxTiles.at<uchar>(tileMapLoc.x, tileMapLoc.y), gyTiles.at<uchar>(tileMapLoc.x, tileMapLoc.y)) * 180 / CV_PI;
                    newrRec = RotatedRect(p, mosaicSize, angleBeta);
                    rSet = positionTile(rRect, tileMap, mapSize, Point(temp.x, temp.y));

                    if (rSet.center.x != -1 && rSet.center.y != -1)
                    {
                        setTile(rSet, image, mosaic);
                        tileMap[tileMapLoc.x][tileMapLoc.y] = rSet; // mark tile, add tile to map
                    }
                }
            }
        }
    }

    //Set remaining tiles
    for (int l = 0; l < numTilesX; l++)
    {
        for (int m = 0; m < numTilesY; m++)
        {
            if (tileMap[m][l].center.x == -1 && tileMap[m][l].center.y == -1) // tile not marked
            {
                Point recCenter(m * mosaicSize.height + tileR, l * mosaicSize.width + tileR);
                angleGamma = atan2(gx.at<uchar>(m, l), gyTiles.at<uchar>(m, l)) * 180 / CV_PI;
                RotatedRect rRect(recCenter, mosaicSize, angleGamma);
                RotatedRect rSet = positionTile(rRect, tileMap, mapSize, Point(m, l));
                if (rSet.center.x != -1 && rSet.center.y != -1)
                {
                    setTile(rSet, image, mosaic);
                    tileMap[m][l] = rSet; // mark tile, add tile to map
                }
            }
        }
    }

    return mosaic;
}

// positionTile - attempts to find a good center for rRect. If no suitable placement found, the function returns rRect at center -1, -1
// Preconditions:
//   - rRect: The RotatedRect to be positioned.
//   - map: A 2D vector representing the map of RotatedRect objects.
//   - mapSize: The size of the map.
//   - mapLoc: The location of the current tile in the map.
// Postconditions:
//   - Returns a RotatedRect object representing the positioned tile. 
//     If a suitable placement is found, the center of the RotatedRect is set to a valid point; 
//     otherwise, the center is set to (-1, -1).

RotatedRect positionTile(RotatedRect rRect, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc)
{
    RotatedRect temp(Point(-1, -1), rRect.size, rRect.angle);
    if (checkPlacementOk(rRect, map, mapSize, mapLoc))
        temp = rRect;
    else
    {
        //Get quadrant to determine shift direction
        //Quadrant indices : 0 | 1
        //                   2 | 3
        int midX = mapSize.width / 2;
        int midY = mapSize.height / 2;
        int quadrant = 0;
        if (mapLoc.x > midX && mapLoc.y < midY)
            quadrant = 1;
        else if (mapLoc.x < midX && mapLoc.y > midY)
            quadrant = 2;
        else if (mapLoc.x > midX && mapLoc.y > midY)
            quadrant = 3;

        Point adjacents[] = { Point(mapLoc.x, mapLoc.y + 1), Point(mapLoc.x, mapLoc.y - 1),
        Point(mapLoc.x + 1, mapLoc.y),  Point(mapLoc.x - 1, mapLoc.y),
        Point(mapLoc.x + 1, mapLoc.y + 1), Point(mapLoc.x - 1, mapLoc.y - 1),
        Point(mapLoc.x - 1, mapLoc.y + 1), Point(mapLoc.x + 1, mapLoc.y - 1) };
        /*
        startingPoint indices:
        0 - Bottom
        1 - Top
        2 - Right
        3 - Left
        4 - Bottom, Right
        5 - Top, Left
        6 - Bottom, Left
        7 - Top, Right
        */
        int startingPoint = 0;
        for (int i = 0; i < 8; i++) //each tile to check for overlap
        {
            //if proposedTile is not on an edge and the adjacent tiles are not empty
            if (adjacents[i].x >= 0 && adjacents[i].x < mapSize.width
                && adjacents[i].y >= 0 && adjacents[i].y < mapSize.height)
            {
                bool mapEmpty = map[adjacents[i].x][adjacents[i].y].center.x == -1
                    && map[adjacents[i].x][adjacents[i].y].center.y == -1; //indicates RotatedRect has not been set
                if (!mapEmpty)
                {
                    vector<Point2f> intersect;
                    RotatedRect r = map[adjacents[i].x][adjacents[i].y];
                    rotatedRectangleIntersection(rRect, r, intersect);
                    if (intersect.size() > 2) //exclude point and line contacts
                    {
                        startingPoint = i;
                        break;
                    }
                }
            }
        }
        temp = shift(quadrant, startingPoint, rRect, map, mapSize, mapLoc);

    }

    return temp;
}

// shift - Shifts the position of rRect within the map based on the specified quadrant and starting point.
// Preconditions:
//   - quadrant: An integer representing the quadrant index (0, 1, 2, or 3) that determines the shift direction.
//   - startingPoint: An integer representing the starting point index (0 to 7) for the shift.
//   - rRect: The RotatedRect to be shifted.
//   - map: A 2D vector representing the map of RotatedRect objects.
//   - mapSize: The size of the map.
//   - mapLoc: The location of the current tile in the map.
// Postconditions:
//   - Returns a RotatedRect object representing the shifted tile. If a suitable placement is found, 
//     the center of the RotatedRect is set to a valid point; otherwise, the center is set to (-1, -1).

RotatedRect shift(int quadrant, int startingPoint, RotatedRect rRect, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc)
{
    RotatedRect temp(Point(-1, -1), rRect.size, rRect.angle);
    //set initial locations
    int R0 = mapLoc.y * rRect.size.height;
    int R1 = mapLoc.y * rRect.size.height + rRect.size.height;
    int C0 = mapLoc.x * rRect.size.width;
    int C1 = mapLoc.x * rRect.size.width + rRect.size.width;

    //if edge of map, don't look
    if (R0 == 0 || C0 == 0 || R1 == mapSize.height || C1 == mapSize.width)
        return temp;
    /*
    0 - Bottom
    1 - Top
    2 - Right
    3 - Left
    4 - Bottom, Right
    5 - Top, Left
    6 - Bottom, Left
    7 - Top, Right
    */

    // Perform different shift patterns based on quadrant and starting point
    if ((startingPoint == 0 && quadrant == 2) || (startingPoint == 2 && quadrant == 0)
        || (startingPoint == 2 && quadrant == 1) || startingPoint == 4)
    { //R-L, B-T
        for (int r = R1; r > R0; r--)
        {
            for (int c = C1; c > C0; c--)
            {
                temp.center.x = c;
                temp.center.y = r;
                if (checkPlacementOk(temp, map, mapSize, mapLoc))
                    return temp;
            }
        }
    }
    else if ((startingPoint == 1 && quadrant == 0) || (startingPoint == 1 && quadrant == 2)
        || (startingPoint == 2 && quadrant == 3) || startingPoint == 7)
    {//R-L, T-B
        for (int c = C0; c < C1; c++)
        {
            for (int r = R1; r > R0; r--)
            {
                temp.center.x = c;
                temp.center.y = r;
                if (checkPlacementOk(temp, map, mapSize, mapLoc))
                    return temp;
            }
        }
    }
    else if ((startingPoint == 3 && quadrant == 2) || (startingPoint == 3 && quadrant == 3)
        || (startingPoint == 1 && quadrant == 3) || startingPoint == 5)
    {//L-R, T-B
        for (int c = C0; c < C1; c++)
        {
            for (int r = R0; r < R1; r++)
            {
                temp.center.x = c;
                temp.center.y = r;
                if (checkPlacementOk(temp, map, mapSize, mapLoc))
                    return temp;
            }
        }
    }
    else if ((startingPoint == 3 && quadrant == 0) || (startingPoint == 3 && quadrant == 1)
        || (startingPoint == 0 && quadrant == 3) || startingPoint == 6)
    {//L-R, B-T
        for (int c = C1; c > C0; c--)
        {
            for (int r = R0; r < R1; r++)
            {
                temp.center.x = c;
                temp.center.y = r;
                if (checkPlacementOk(temp, map, mapSize, mapLoc))
                    return temp;
            }
        }
    }
    else
    {
        temp.center.x = -1;
        temp.center.y = -1;
    }
    temp.center.x = -1;
    temp.center.y = -1;
    return temp;
}

// getPerpendicularTile - Calculates the center point for a rotatedRect based on the perpendicular direction (90 or -90) 
//                        from the given angle.
// Preconditions:
//   - rRect: The RotatedRect for which the perpendicular tile center is calculated.
//   - direction: An integer representing the perpendicular direction (-90 or 90).
//   - s: The size of the image or area where the tile will be placed.
//   - angleAlpha: The angle in degrees for the perpendicular direction.
// Postconditions:
//   - Returns a Point object representing the center point of the perpendicular tile. 
//     If a suitable point is not found (out of bounds or overlaps with the original tile), it returns the default point (-1, -1).

Point getPerpendicularTile(RotatedRect rRect, int direction, Size s, float angleAlpha)
{
    Point p(-1, -1);
    cv::Point2f pt1, pt2;
    float theta = (angleAlpha)*CV_PI / 180.0; // Convert angle to radians

    if (direction == -90)
    {
        p.x = rRect.center.x - rRect.size.height * cos(theta);
        p.y = rRect.center.y - rRect.size.width * sin(theta);
    }
    else
    {
        p.x = rRect.center.x + rRect.size.height * cos(theta);
        p.y = rRect.center.y + rRect.size.width * sin(theta);
    }

    //Check if the point is valid for the new tile
    if (p.x < 0 || p.y < 0 || p.x >= s.width || p.y >= s.height)
    {
        p.x = -1;
        p.y = -1;
        return p;
    }

    //Adjust the center point based on tile size
    p.x = (p.x / int(rRect.size.height)) * rRect.size.height + rRect.size.height / 2;
    p.y = p.y / int(rRect.size.width) * rRect.size.width + rRect.size.width / 2;

    //Check if the adjusted center point overlaps with the original tile or goes beyond the image boundaries
    if (p.x == rRect.center.x && p.y == rRect.center.y ||
        p.x + rRect.size.height > s.width || p.y + rRect.size.width > s.height)
    {
        p.x = -1; p.y = -1;
    }

    return p;
}


// setTile - Gets the tile area of the original image, finds the average color within that area,
//           and fills the corresponding rotated rectangle with the average color on the output image (mosaic).
// Preconditions:
//   - rRect: The RotatedRect specifying the area of the original image to be considered as a tile.
//   - image: The original image from which the tile area is extracted.
//   - mosaic: The output image (mosaic) where the filled rotated rectangle will be placed.
// Postconditions: n/a
void setTile(RotatedRect rRect, Mat image, Mat mosaic)
{
    Point2f tl(rRect.center.x - rRect.size.width / 2.0, rRect.center.y - rRect.size.height / 2.0);
    Rect rec(tl, rRect.size);
    Point vertices[4];
    getVertices(vertices, rRect);
    Mat tile = image(rec);
    Scalar avgColor = mean(tile);
    fillConvexPoly(mosaic, vertices, 4, avgColor);
}


// getAvgVal - Calculates the average luminance of an image.
// Preconditions:
//   - tile: The input image should be in grayscale format for which the average luminance is calculated.
// Postconditions:
//   - Returns the average luminance as a floating-point value.
float getAvgVal(Mat tile)
{
    float avg = 0;
    int totalLuminance = 0;
    for (int tx = 0; tx < tile.rows; tx++)
    {
        for (int ty = 0; ty < tile.cols; ty++)
            totalLuminance += (int)tile.at<uchar>(tx, ty);
    }
    avg = totalLuminance / (tile.rows * tile.cols);
    return avg;
}

// getVertices - Retrieves the vertices of a RotatedRect and stores them in a Point array.
// Preconditions:
//   - arr: An array of four Point objects that will store the vertices of the rotated rectangle.
//   - rRect: The RotatedRect from which to extract the vertices.
// Postconditions:
//   - The vertices of the RotatedRect are stored in the Point array arr[4] after the function runs.
void getVertices(Point* arr, RotatedRect rRect)
{
    Point2f vertices2f[4];
    rRect.points(vertices2f);
    for (int i = 0; i < 4; ++i) {
        arr[i] = vertices2f[i];
    }
}


// checkPlacementOk - Checks if the RotatedRect will intersect with other RotatedRects on the tileMap.
// Preconditions:
//   - proposedTile: The RotatedRect to be checked for placement.
//   - map: A 2D vector representing the tileMap that contains other RotatedRects.
//          It is passed by reference since the map can be modified.
//   - mapSize: The size of the tileMap.
//   - mapLoc: The location of the proposedTile on the tileMap.
// Postconditions:
//   - Returns true if the proposedTile can be placed without intersecting with other RotatedRects on the tileMap.

bool checkPlacementOk(RotatedRect proposedTile, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc)
{
    Point center = proposedTile.center;
    Point vertices[4];
    int size = proposedTile.size.height; //assumes tiles are square
    getVertices(vertices, proposedTile);
    Point adjacents[] = { Point(mapLoc.x, mapLoc.y + 1), Point(mapLoc.x, mapLoc.y - 1),
        Point(mapLoc.x + 1, mapLoc.y),  Point(mapLoc.x - 1, mapLoc.y),
        Point(mapLoc.x + 1, mapLoc.y + 1), Point(mapLoc.x - 1, mapLoc.y - 1),
        Point(mapLoc.x - 1, mapLoc.y + 1), Point(mapLoc.x + 1, mapLoc.y - 1) };

    for (int i = 0; i < 8; i++) //check each adjacent tile for overlap
    {
        if (adjacents[i].x >= 0 && adjacents[i].x < mapSize.width
            && adjacents[i].y >= 0 && adjacents[i].y < mapSize.height)
        {
            //if proposedTile is not on an edge and the adjacent tiles are not empty
            bool mapEmpty = map[adjacents[i].x][adjacents[i].y].center.x == -1
                && map[adjacents[i].x][adjacents[i].y].center.y == -1;
            if (!mapEmpty)
            {
                vector<Point2f> intersect;
                RotatedRect r = map[adjacents[i].x][adjacents[i].y];
                rotatedRectangleIntersection(proposedTile, r, intersect);
                if (intersect.size() > 1) //exclude point contacts
                    return false;
            }
        }
    }

    return true; //ok to lay the tile without overlap
}
