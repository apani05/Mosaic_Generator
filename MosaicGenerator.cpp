//MosaicGenerator.cpp - inputs image and outputs an image with a mosaic tranformation. 
// Aishwarya Pani and Tabitha Roemish

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
//#include <opencv2/features2d.hpp>
#include <iostream>
enum { red, green, blue };
using namespace cv;
using namespace std;
Mat calcLuminance(Mat image);
Mat calcGradient(Mat image);
Mat calcGVFField(Mat image, Mat gx, Mat gy);
Mat calcNonMax(Mat image);
Mat placeTiles(Mat image, Mat nonMax, Mat gx, Mat gy, int tileSize);
//helper methods
void setTile(RotatedRect rRect, Mat image, Mat mosaic);
RotatedRect positionTile(RotatedRect rRect, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc);
RotatedRect shift(int quadrant, int startingPoint, RotatedRect rRect, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc);
bool checkPlacementOk(RotatedRect tile, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc);
Point getPerpendicularTile(RotatedRect rRect, int direction, Size s, float angleAlpha);
void displayImage(string name, Mat image);
float getAvgVal(Mat tile);
void getVertices(Point* arr, RotatedRect rRect);


//main - main program reads in an image and outputs a mosaic-ized image
//preconditions: image must be in same directory
//postconditions: image is saved in same directory as code
int main(int argc, char* argv[]) {

    // Step 1: Read an input image from directory
    Mat image = imread("lena.jpg"); //VAnne.jpg, sqrl.jpg
    resize(image, image, Size(667, 667)); //for lena image

    // Step 2: Calculate the luminance of the image
    Mat luminance = calcLuminance(image);

    // Step 3: Calculate Robert's gradient of the luminance image 
    Mat robert = calcGradient(luminance);

    // Step 4 - 5: Calculate Gradient Vector Flow Map (might need size)
    Mat gx = Mat::zeros(image.size(), CV_8UC1);
    Mat gy = Mat::zeros(image.size(), CV_8UC1);
    Mat gvf = calcGVFField(robert, gx, gy);

    // Step 6: Calculate NonMaximumSuppression
    Mat nonMax = calcNonMax(gvf);

    // Step 7-23: Calculate tile Angles and place tiles  
    int const tilesize = 4; // 1 pixel X 1 pixel
    Mat mosaic = placeTiles(image, nonMax, gx, gy, tilesize);

    // Display Images
    displayImage("Input Image", image);
    displayImage("Luminance", luminance);
    displayImage("Gradient", robert);
    displayImage("GVF", gvf);
    displayImage("nonMax", nonMax);
    displayImage("mosaic", mosaic);
    waitKey(0);
    // save image and end
    imwrite("MosaicFinalImage.jpg", mosaic);
    return 0;
}

void displayImage(string name, Mat image)
{
    namedWindow(name, WINDOW_NORMAL);
    resizeWindow(name, image.cols, image.rows);
    imshow(name, image);
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
Mat calcGVFField(Mat image, Mat gx, Mat gy) {
    Mat gvf = Mat::zeros(image.size(), CV_8UC1);
    int ddepth = CV_8UC1;
    int scale = 1;
    int delta = 0;
    Mat mygx = Mat::zeros(image.size(), CV_8UC3);
    Mat mygy = Mat::zeros(image.size(), CV_8UC3);
    Scharr(image, gx, ddepth, 1, 0, scale);
    Scharr(image, gy, ddepth, 0, 1, scale);


    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            Scalar xval = gx.at<uchar>(i, j); // Notice <uchar> not float
            Scalar yval = gy.at<uchar>(i, j);
            gvf.at<uchar>(i, j) = (xval.val[0], yval.val[0]);
        }
    }
    //displayImage("gx", gx);
    //displayImage("gy", gy);
    //displayImage("gvf", gvf);
    //waitKey(0);

    return gvf;
}

//calcNonMax - calculates the non-maximum suppression for each tile
//preconditions:
//postconditions:
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

//placeTiles - this final method creates the final mosaic image by placing the tiles according to the angleMap
//preconditions:
//postconditions:
Mat placeTiles(Mat image, Mat nonMax, Mat gx, Mat gy, int tileSize) {

    float avgLum = getAvgVal(nonMax);
    int threshold_h = avgLum * .80;
    int threshold_l = avgLum * .20;
    Mat mosaic(Size(image.cols, image.rows), image.type(), cv::Scalar(102.0, 102.0, 102.0)); //dark slate grey
    queue<Point> Q;
    float angleAlpha = 0;
    float angleBeta = 0;
    float angleGamma = 0;
    //-------------------
    Size mosaicSize(tileSize, tileSize);  // Set the desired mosaic size
    float tileR = tileSize / 2;
    int numTilesX = image.cols / mosaicSize.width;
    int numTilesY = image.rows / mosaicSize.height;

    //divide gx and gy into tiles
    Mat gxTiles = Mat::zeros(Size(numTilesX, numTilesY), CV_8UC1);
    Mat gyTiles = Mat::zeros(Size(numTilesX, numTilesY), CV_8UC1);

    //Set up map of RotatedRects to check for overlap
    RotatedRect emptyCell(Point2f(-1, -1), mosaicSize, 0);
    std::vector<std::vector<RotatedRect> > tileMap(numTilesX, std::vector<RotatedRect>(numTilesY, emptyCell));
    Size mapSize(tileMap.size(), tileMap[0].size());

    //fill queue & sort Queue
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
            //fill gxTiles and gyTiles with average change
            float avgXChange = getAvgVal(gxTile);
            float avgYChange = getAvgVal(gyTile);
            gxTiles.at<uchar>(i, j) = avgXChange;
            gyTiles.at<uchar>(i, j) = avgYChange;
        }
    }

    //sort vector from largest gradient to smallest
    sort(v.begin(), v.end(),
        [](const tuple<float, Point>& a,
            const tuple<float, Point>& b) -> bool
        {
            return std::get<0>(a) > std::get<0>(b);
        });

    for (int i = 0; i < v.size(); i++)
    {
        Q.push(get<1>(v[i]));
    }

    //start placing edge and adjacent tiles
    while (!Q.empty())
    {
        Point temp = Q.front();
        Q.pop();
        if (tileMap[temp.x][temp.y].center.x == -1 && tileMap[temp.x][temp.y].center.y == -1)
        {
            //set tiles > threshold_h 
            Point recCenter(temp.x * mosaicSize.width + tileR, temp.y * mosaicSize.height + tileR);
            angleAlpha = atan2(gyTiles.at<uchar>(temp.x, temp.y), gxTiles.at<uchar>(temp.x, temp.y)) * (180 / CV_PI);
            RotatedRect rRect(recCenter, mosaicSize, angleAlpha); //initial tile
            RotatedRect rSet = positionTile(rRect, tileMap, mapSize, Point(temp.x, temp.y)); //shifted tile
            //if (checkPlacementOk(rRect, tileMap, mapSize, Point(temp.x, temp.y)))
            if (rSet.center.x != -1 && rSet.center.y != -1)
            {
                setTile(rSet, image, mosaic);
                tileMap[temp.x][temp.y] = rSet; // add tile to map
                //displayImage("mosaicInProg", mosaic);
                //waitKey(0);
            }
            //set tiles >  threshold_l 
            int direction = 90;
            RotatedRect newrRec = rRect;
            while (true)
            {//start looking for adjacent tiles perpendicular to edge tile
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
                { //tile less than threshold_l or tile is already set
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
                    //if (checkPlacementOk(newrRec, tileMap, mapSize, tileMapLoc))
                    if (rSet.center.x != -1 && rSet.center.y != -1)
                    {
                        setTile(rSet, image, mosaic);
                        tileMap[tileMapLoc.x][tileMapLoc.y] = rSet; // mark tile, add tile to map
                        //displayImage("mosaicInProg", mosaic);
                        //waitKey(0);
                    }
                }
            }
        }
    }
    //displayImage("mosaicInProg", mosaic);
    //waitKey(0);
   //set remaining tiles
    for (int l = 0; l < numTilesX; l++)
    {
        for (int m = 0; m < numTilesY; m++)
        {
            if (tileMap[m][l].center.x == -1 && tileMap[m][l].center.y == -1) // not marked
            {
                Point recCenter(m * mosaicSize.height + tileR, l * mosaicSize.width + tileR);
                angleGamma = atan2(gx.at<uchar>(m, l), gyTiles.at<uchar>(m, l)) * 180 / CV_PI;
                RotatedRect rRect(recCenter, mosaicSize, angleGamma);
                RotatedRect rSet = positionTile(rRect, tileMap, mapSize, Point(m, l));
                //if (checkPlacementOk(rRect, tileMap, mapSize, Point(m, l)))
                if (rSet.center.x != -1 && rSet.center.y != -1)
                {
                    setTile(rSet, image, mosaic);
                    tileMap[m][l] = rSet; // mark tile, add tile to map
                    //displayImage("mosaicInProg", mosaic);
                    //waitKey(0);
                }
            }
        }
    }

    return mosaic;
}

//positionTile - attempts to find a good center for rRect. If no suitable placement found
// function returns rRect at center -1, -1
RotatedRect positionTile(RotatedRect rRect, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc)
{
    RotatedRect temp(Point(-1, -1), rRect.size, rRect.angle);
    if (checkPlacementOk(rRect, map, mapSize, mapLoc))
        temp = rRect;
    else
    {
        //Get quadrant to determine shift direction
        // 0 | 1
        // 2 | 3
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



//getPerpendicularTile - rotatedRect and returns center point based on 
// the perpendicular direction (90 or -90)
//preconditions - 
//postconditions - returns the default point at (-1, -1) if there is not tile in the specified direction
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

    //check point is valid for new tile
    if (p.x < 0 || p.y < 0 || p.x >= s.width || p.y >= s.height)
    {
        p.x = -1;
        p.y = -1;
        return p;
    }
    //get center point
    p.x = (p.x / int(rRect.size.height)) * rRect.size.height + rRect.size.height / 2;
    p.y = p.y / int(rRect.size.width) * rRect.size.width + rRect.size.width / 2;
    if (p.x == rRect.center.x && p.y == rRect.center.y ||
        p.x + rRect.size.height > s.width || p.y + rRect.size.width > s.height)
    {
        p.x = -1; p.y = -1;
    }


    return p;
}

//setTile - this method gets the tile area of the original image, finds the average color,
    //then fills the corresponding rotated rect with the average color and places it on the output image
//preconditions - requires Rect and RotatedRect to be created at specific area of image. 
//postconditions - n/a
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



//getAveLum - gets the average luminance of an image
//preconditions: image should be grayscale
//postconditions: returns average luminance in float
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

//getVertices - fills a Point array of 4 with vertices from roatedRect
//preconditions - requires Point arr[4] to be created
//postconitions - Point arr[4] contains the vertices after method runs
void getVertices(Point* arr, RotatedRect rRect)
{
    Point2f vertices2f[4];
    rRect.points(vertices2f);
    for (int i = 0; i < 4; ++i) {
        arr[i] = vertices2f[i];
    }
}

//checkOverlap - checks to see if the rotatedRect will intersect with other RotatedRects on the tileMap
//preconditions - needs tileMap and map size since tileMap is passed by reference
//postconditions - returns true if ok to place tile
bool checkPlacementOk(RotatedRect proposedTile, vector<vector<RotatedRect>>& map, Size mapSize, Point mapLoc) {

    //vector<Point> adjacentTiles;
    Point center = proposedTile.center;
    Point vertices[4];
    int size = proposedTile.size.height; //assumes tiles are square
    getVertices(vertices, proposedTile);
    Point adjacents[] = { Point(mapLoc.x, mapLoc.y + 1), Point(mapLoc.x, mapLoc.y - 1),
        Point(mapLoc.x + 1, mapLoc.y),  Point(mapLoc.x - 1, mapLoc.y),
        Point(mapLoc.x + 1, mapLoc.y + 1), Point(mapLoc.x - 1, mapLoc.y - 1),
        Point(mapLoc.x - 1, mapLoc.y + 1), Point(mapLoc.x + 1, mapLoc.y - 1) };

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
                rotatedRectangleIntersection(proposedTile, r, intersect);
                if (intersect.size() > 1) //exclude point contacts
                    return false;
            }
        }
    }

    return true; // ok to lay tile
}



