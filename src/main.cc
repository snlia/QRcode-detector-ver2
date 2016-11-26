#include <iostream>
#include <algorithm>
#include <queue>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define maxn 2000

using namespace std;
using namespace cv;

double pickRatio = 0.5;

Mat frame, bin, rawFrame;
int upN [maxn][maxn], rightN [maxn][maxn], downN [maxn][maxn], leftN [maxn][maxn];
deque<Point> q;
int f [maxn][maxn];
int area [maxn * maxn];

int parameter_init (int argc, const char *argv[]) {
    return 0;
}

bool ckRatio (int len1, int len2, int len3, int len4, int len5) {
    if (abs (len1 - len2) > len1 * pickRatio) return 0;
    if (abs (len1 * 3 - len3) > len1 * pickRatio * 3) return 0;
    if (abs (len1 - len4) > len1 * pickRatio) return 0;
    if (abs (len1 - len5) > len1 * pickRatio) return 0;
    return 1;
}

void fillarea (int x, int y) {
    q.clear (); 
    q.push_back (Point (x, y));
    vector<Point> go;
    go.clear ();
    go.push_back (Point (0, -1)); go.push_back (Point (0, 1));
    go.push_back (Point (1, 0)); go.push_back (Point (-1, 0));
    go.push_back (Point (1, -1)); go.push_back (Point (-1, 1));
    go.push_back (Point (1, 1)); go.push_back (Point (-1, -1));
    for (Point nowX; q.size (); q.pop_front ()) {
        nowX = q[0];
        circle (frame, cvPoint (q[0].x, q[0].y), 2, Scalar(255,0,0), 2, 8, 0);
        return ;
        for (int i = 0; i < 8; ++i) {
            Point to = nowX + go[i];
            if (to.x && to.y && to.x < frame.cols && to.y < frame.rows && 
                    bin.at<uchar>(to.y, to.x) == bin.at<uchar>(y, x)) {
                q.push_back (to);
            }
        }
    }
}

bool ckArea (int area1, int area2, int area3) {
    if (abs (area1 * 2 - area2 * 3) > area1 * 2 * 0.5) return 0;
    if (abs (area1 * 3 - area3 * 8) > area1 * 3 * 0.5) return 0;
    return 1;
}

void markFIPdown (int x, int y) {
    int len1 = downN[x][y] - y;
    int y2 = downN[x][y];
    if (y2 >= frame.rows) return;
    int len2 = downN[x][y2] - y2;
    int y3 = downN[x][y2];
    if (y3 >= frame.rows) return;
    int len3 = downN[x][y3] - y3;
    int y4 = downN[x][y3];
    if (y4 >= frame.rows) return;
    int len4 = downN[x][y4] - y4;
    int y5 = downN[x][y4];
    if (y5 >= frame.rows) return;
    int len5 = downN[x][y5] - y5;
    if (ckRatio (len1, len2, len3, len4, len5)) {
        printf ("%d %d %d %d %d\n", area[f[x][y]], area[f[x][y3]], area[f[x][y4]], area[f[x][y5]]);
        if (ckArea (area[f[x][y]], area[f[x][y2]], area[f[x][y3]])) {
            fillarea (x, y);
            fillarea (x, y2);
            fillarea (x, y3);
            fillarea (x, y4);
            fillarea (x, y5);
        }
    }
}

void markFIPright (int x, int y) {
    int len1 = rightN[x][y] - x;
    int x2 = rightN[x][y];
    if (x2 >= frame.cols) return;
    int len2 = rightN[x2][y] - x2;
    int x3 = rightN[x2][y];
    if (x3 >= frame.rows) return;
    int len3 = rightN[x3][y] - x3;
    int x4 = rightN[x3][y];
    if (x4 >= frame.rows) return;
    int len4 = rightN[x4][y] - x4;
    int x5 = rightN[x4][y];
    if (x4 >= frame.rows) return;
    int len5 = rightN[x5][y] - x5;
    if (ckRatio (len1, len2, len3, len4, len5)) {
        if (ckArea (area[f[x][y]], area[f[x2][y]], area[f[x3][y]])) {
            fillarea (x, y);
            fillarea (x2, y);
            fillarea (x3, y);
            fillarea (x4, y);
            fillarea (x5, y);
        }
    }
}

void MarkLine () {

    for (int i = 0; i < bin.cols; ++i) 
        for (int j = 0; j < bin.rows; ++j) {
            upN[i][j] = rightN[i][j] = downN[i][j] = leftN[i][j] = 0;
        }

    // Caculate the size of continuous block
    // Downwards
    for (int i = 0; i < bin.cols; ++i) 
        for (int j = 0, k = 0; j < bin.rows; downN[i][j] = k, j = k)
            for (k = j; k < bin.rows && bin.at<uchar> (k, i) == bin.at<uchar> (j, i); ++k);

    // Rightwards
    for (int i = 0; i < bin.rows; ++i) 
        for (int j = 0, k = 0; j < bin.cols; rightN[j][i] = k, j = k)
            for (k = j; k < bin.cols && bin.at<uchar> (i, k) == bin.at<uchar> (i, j); ++k);

    // Mark the FIP
    // Downwards
    for (int i = 0; i < bin.cols; ++i)
        for (int j = 0; j < bin.rows; j = downN[i][j]) 
            markFIPdown (i, j);

    // Rightwards
    for (int i = 0; i < bin.rows; ++i)
        for (int j = 0; j < bin.cols; j = rightN[j][i]) 
            markFIPright (j, i);
}

void bfs (int x, int y, int cnt) {
    int size = 0;
    q.clear ();
    q.push_back (Point (x, y));
    vector<Point> go;
    go.clear ();
    go.push_back (Point (0, -1)); go.push_back (Point (0, 1));
    go.push_back (Point (1, 0)); go.push_back (Point (-1, 0));
    go.push_back (Point (1, -1)); go.push_back (Point (-1, 1));
    go.push_back (Point (1, 1)); go.push_back (Point (-1, -1));
    area[cnt] = 1;
    for (Point nowX; q.size (); q.pop_front ()) {
        nowX = q[0];
        for (int i = 0; i < 8; ++i) {
            Point to = nowX + go[i];
            if (to.x && to.y && to.x < frame.cols && to.y < frame.rows && 
                    !~f[to.x][to.y] && bin.at<uchar>(to.y, to.x) == bin.at<uchar>(y, x)) {
                f[to.x][to.y] = cnt;
                ++area[cnt];
                q.push_back (to);
            }
        }
    }
}

void floodfill (void) {
    for (int i = 0; i < bin.cols; ++i)
        for (int j = 0; j < bin.rows; ++j) f[i][j] = -1;

    int cnt = 0;
    vector<Point> go;
    go.clear (); go.push_back (Point (-1, 0));
    go.push_back (Point (0, -1)); go.push_back (Point (-1, -1));
    for (int i = 0; i < bin.cols; ++i)
        for (int j = 0; j < bin.rows; ++j) 
            if (!~f[i][j]) bfs (i, j, ++cnt);
}

int main(int argc, const char *argv[]) {
    
    // Set up parameters
    if (parameter_init (argc, argv)) return 0;

    VideoCapture capture(0);

    Mat gray(frame.size(), CV_MAKETYPE(frame.depth(), 1));
    Mat marked(frame.size(), CV_MAKETYPE(frame.depth(), 1));
    Mat detected_edges(frame.size(), CV_MAKETYPE(frame.depth(), 1));
    capture >> frame;
//    frame = imread ("../data/11.jpg");

    cout << "Press any key to return." << endl;

    for (int key = -1; !~key; capture >> frame) {
        frame.copyTo (rawFrame);
        // Change to grayscale
        cvtColor (rawFrame, gray, CV_RGB2GRAY);
        threshold (gray, bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        floodfill ();

        // Find FIP candidates
        MarkLine ();

        imshow ("Image", frame);
        imshow ("Bin", bin);
        //imshow ("Marked", marked);

        key = waitKey (100);
    }

    return 0;
}
