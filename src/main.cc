#include <iostream>
#include <algorithm>
#include <queue>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "swordfeng.h"
#include <boost/program_options.hpp>

#define maxn 2000
#define sqr(x) ((x) * (x))
#define INF 10000000
#define pause for (int key = -1; !~key; key = waitKey (100))
#define printc(x,y,z) circle (frame, cvPoint ((x), (y)), 1 , (z) == 0 ? Scalar(255,0,0) : ((z) == 1 ? Scalar(0,255,0) : Scalar(0,0,255)), 2, 8, 0);


using namespace std;
using namespace cv;
namespace po = boost::program_options;

double pickRatioL = 0.5;
double pickRatioA = 0.7;
bool useimage = 0;
int hierarchythre = 4;
int qrsize = 100;
double areathre = 0.005;
double distthre = 0.2;

const Point go [8] = {Point(0, 1), Point(1, 0), Point(0, -1), Point(-1, 0),
Point(1, 1), Point(1, -1), Point(-1, 1), Point(-1, -1)};

Mat frame, rawFrame, bin;
int upN [maxn][maxn], rightN [maxn][maxn], downN [maxn][maxn], leftN [maxn][maxn];
deque<Point> q;
bool v [maxn][maxn], vhull [maxn][maxn];
int f [maxn][maxn];
vector<vector<Point>> FIP;
int Cid [maxn][maxn];
vector<vector<Point>> candidates;
int area [maxn * maxn];

extern void LocalThBinarization (Mat qr, Mat &out);
extern void LocalPreWorkGray (Mat &qrGray);


double dist (Point2f x, Point2f y) {
    // Return the distance between two points
    return sqrt (sqr (x.x - y.x) + sqr (x.y - y.y));
}

double crossProduct (Point2f A, Point2f B, Point2f C) {
    return ((B.x - A.x) * (C.y - B.y) - (C.x - B.x) * (B.y - A.y));
}

double distLine (Point2f X, Point2f A, Point2f B) {
    // Returen the distance between Point X and Line AB
    return abs (crossProduct (X, A, B) / dist (A, B));
}

double dotProduct (Point2f A, Point2f B) {
    return A.x * B.x + A.y * B.y;
}

double cosAngle (Point2f A, Point2f B, Point2f C, Point2f D) {
    return (dotProduct (Point2f (B.x - A.x, B.y - A.y), Point2f (D.x - C.x, D.y - C.y)) /
            (dist (A, B) * dist (C, D)));
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
Point2f intersection (Point2f o1, Point2f p1, Point2f o2, Point2f p2) {
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;

    return o1 + d1 * t1;
}

vector<int> getPoint (double AB, double BC, double CA, int A, int B, int C) {
    vector<int> res;
    res.clear ();
    if (AB > BC && AB > CA) {
        res.push_back (C);
        res.push_back (A);
        res.push_back (B);
    }
    if (BC > AB && BC > CA) {
        res.push_back (A);
        res.push_back (B);
        res.push_back (C);
    }
    if (CA > AB && CA > BC) {
        res.push_back (B);
        res.push_back (A);
        res.push_back (C);
    }
    return res;
}

bool dist_constraint (double AB, double BC, double CA) {
    // TODO : add more constraint
    if (AB > BC && AB > CA) {
        if (abs (BC - CA) > distthre * max (BC, CA)) return 1;
        return 0;
    }
    if (BC > AB && BC > CA) {
        if (abs (AB - CA) > distthre * max (CA, AB)) return 1;
        return 0;
    }
    if (CA > AB && CA > BC) {
        if (abs (BC - AB) > distthre * max (BC, AB)) return 1;
        return 0;
    }
    return 1;
}

bool area_constraint (double areaA, double areaB, double areaC) {
    double areaMean = (areaA + areaB + areaC) / 3;
    double sigma = sqr (areaA - areaMean) + sqr (areaB - areaMean) + sqr (areaC - areaMean);
    sigma /= 3 * sqr (areaMean);
    return sigma > areathre;
}

Point2f findAwayFromLine (int x, Point2f A, Point2f B) {
    // Find a Point in FIP[x] that most away from line AB 
    Point2f res;
    double maxP = 0;
    for (int i = 0; i < FIP[x].size(); ++i) {
        double tmp = distLine (FIP[x][i], A, B);
        if (tmp > maxP) {
            maxP = tmp;
            res = FIP[x][i];
        } 
    }
    return res;
}

Point2f findAwayFromPoint (int x, Point2f P) {
    // Find a Point in FIP[x] that most away from point P
    Point2f res;
    double maxP = 0;
    for (int i = 0; i < FIP[x].size(); ++i) {
        double tmp = dist (FIP[x][i], P);
        if (tmp > maxP) {
            maxP = tmp;
            res = FIP[x][i];
        } 
    }
    return res;
}

Point2f findN (Point2f P1, Point2f P2, Point2f P4, int top, int left, int right) {
    Point2f verP1 = findAwayFromPoint (top, P1);

    // Find the right aligned point for corner P2
    Point2f verP2;
    double maxTheta = -INF;
    for (int i = 0; i < FIP[right].size (); ++i) {
        double tmp = cosAngle (P1, verP1, P2, FIP[right][i]);
        if (tmp > maxTheta) {
            maxTheta = tmp;
            verP2 = FIP[right][i];
        }
    }
    // Find the left aligned point for corner P4
    Point2f verP4;
    maxTheta = -INF;
    for (int i = 0; i < FIP[left].size (); ++i) {
        double tmp = cosAngle (P1, verP1, P4, FIP[left][i]);
        if (tmp > maxTheta) {
            maxTheta = tmp;
            verP4 = FIP[left][i];
        }
    }

    // Caculate the intersection of P1verP1 and P2verP2, we use the result to do the first perspective transform
    Point2f P3, originP3;
/*    if (dist (P1, intersection (P1, verP1, P2, verP2)) > dist (P1, intersection (P1, verP1, P4, verP4)))
        originP3 = P3 = intersection (P1, verP1, P2, verP2);
    else
        originP3 = P3 = intersection (P1, verP1, P4, verP4);
        */
    originP3 = intersection (P2, verP2, P4, verP4);

    // relocate P3
    vector<Point2f> pts1, pts2;
    Mat M, gray, firstImg, bin;
    double ratio, optRatio = 0;
    int minFlag = INF;
    for (ratio = -0.2; ratio < 0.1; ratio += 0.01) {
        P3 = originP3 + (verP1 - P1) * ratio;
        pts1.clear (); pts2.clear ();
        pts1.push_back (P1); pts1.push_back (P2);
        pts1.push_back (P3); pts1.push_back (P4);
        pts2.push_back (Point2f (0, 0));
        pts2.push_back (Point2f (qrsize, 0));
        pts2.push_back (Point2f (qrsize, qrsize));
        pts2.push_back (Point2f (0, qrsize));
        M = getPerspectiveTransform (pts1, pts2);
        warpPerspective (rawFrame, firstImg, M, Size (qrsize, qrsize));
        // convert to binary image
        cvtColor (firstImg, gray,CV_RGB2GRAY);
        LocalPreWorkGray (gray);
        //    threshold (gray, bin, 180, 255, CV_THRESH_BINARY);
        LocalThBinarization (gray, bin);
        int flag = 0;
        for (int i = int (qrsize * 0.9); i < qrsize; ++i) {
            flag += bin.at<uchar>(qrsize - 1, i) == 0;
            flag += bin.at<uchar>(i, qrsize - 1) == 0;
        }
        /*
        if (minFlag == INF) {
            imshow ("bin", bin);
            pause;
        }*/
        if (flag <= minFlag) {
            minFlag = flag;
            optRatio = ratio;
        }
   //     printf ("%d\n", flag);
  //      printf ("%.4lf\n", ratio);
    }
//    printf ("OPT %.4lf\n", optRatio);

    // Caculate the optium P3
    P3 = originP3 + (verP1 - P1) * optRatio;
    pts1.clear (); pts2.clear ();
    pts1.push_back (P1); pts1.push_back (P2);
    pts1.push_back (P3); pts1.push_back (P4);
    pts2.push_back (Point2f (0, 0));
    pts2.push_back (Point2f (qrsize, 0));
    pts2.push_back (Point2f (qrsize, qrsize));
    pts2.push_back (Point2f (0, qrsize));
    M = getPerspectiveTransform (pts1, pts2);
    warpPerspective (rawFrame, firstImg, M, Size (qrsize, qrsize));
    // convert to binary image
    cvtColor (firstImg, gray,CV_RGB2GRAY);
    LocalPreWorkGray (gray);
    //    threshold (gray, bin, 180, 255, CV_THRESH_BINARY);
    LocalThBinarization (gray, bin);

    //imshow ("bin", bin);
    //pause;

    // Find K1, K2
    vector<Point2f> K1, K2;
    K1.push_back (Point2f(qrsize - 1, qrsize - 1));
    K2.push_back (Point2f(qrsize - 1, qrsize - 1));
    double minK1 = INF, minK2 = INF;
    for (int i = qrsize - 1; i > int (0.7 * qrsize); --i) 
        for (int j = qrsize - 1; j > int (0.7 * qrsize); --j)
            if (bin.at<uchar>(j, i) == 0) {
                // Update K1
                if (minK1 > (qrsize - i + 0.0) / j) {
                    minK1 = (qrsize - i + 0.0) / j;
                    K1[0] = Point2f(i, j);
                }
                // Update K2
                if (minK2 > (qrsize - j + 0.0) / i) {
                    minK2 = (qrsize - j + 0.0) / i;
                    K2[0] = Point2f(i, j);
                }
            }
    // P3 should be the intersection of P4K2 and P2K1
    M = getPerspectiveTransform (pts2, pts1);
    vector<Point2f> realK1, realK2;
    perspectiveTransform (K1, realK1, M);
    perspectiveTransform (K2, realK2, M);
    return intersection (P4, realK2[0], P2, realK1[0]);
}

vector<Point2f> findCorners (int top, int left, int right, Point2f meanTop, Point2f meanLeft, Point2f meanRight) {
    vector<Point2f> res;
    res.clear ();
    Point2f A = findAwayFromLine (top, meanLeft, meanRight);
    Point2f verA = findAwayFromPoint (top, A);
    Point2f B = findAwayFromLine (right, A, verA);
    Point2f C = findAwayFromLine (left, A, verA);
    res.push_back (A);
    res.push_back (B);
    res.push_back (C);
    res.push_back (findN (A, B, C, top, left, right));
    return res;
}

void findQR (Mat &qr, bool &flag) {
    Mat raw;
    qr = Mat::zeros(qrsize, qrsize, CV_8UC1);
    raw = Mat::zeros(qrsize, qrsize, CV_8UC1);
    int size = FIP.size ();
    if (size < 3) {
        flag = 0;
        return;
    };
    vector<Moments> mome(size);
    vector<Point2f> mean(size);
    vector<Point2f> pts1, pts2;
    // Caculate the mean point
    for (int i = 0; i < size; ++i) {
        mome[i] = moments (FIP[i], false); 
        mean[i] = Point2f (mome[i].m10/mome[i].m00 , mome[i].m01/mome[i].m00);
    }
    for (int A = 0; A < size; ++A) 
        for (int B = A + 1; B < size; ++B)
            for (int C = B + 1; C < size; ++C) {
                double AB = dist (mean[A],mean[B]);
                double BC = dist (mean[B],mean[C]);
                double CA = dist (mean[C],mean[A]);
                if (dist_constraint (AB, BC, CA)) continue;
                if (area_constraint (contourArea (FIP[A]), 
                            contourArea (FIP[B]), 
                            contourArea (FIP[C]))
                   ) 
                    continue;
                vector<int> tmp = getPoint (AB, BC, CA, A, B, C);
                int top = tmp[0]; int left = tmp[1]; int right = tmp[2];
                // Use cross product to determine left and right
                if (crossProduct (mean[left], mean[top], mean[right]) < 0) 
                    swap (left, right);
                // Find all corners
                pts1 = findCorners (top, left, right, mean[top], mean[left], mean[right]);
                /*
                   circle (frame, cvPoint (pts1[0].x, pts1[0].y), 10 , Scalar(255,0,0), 2, 8, 0);
                   circle (frame, cvPoint (pts1[2].x, pts1[2].y), 10, Scalar(0,255,0), 2, 8, 0);
                   circle (frame, cvPoint (pts1[1].x, pts1[1].y), 10, Scalar(0,0,255), 2, 8, 0);
                   circle (frame, cvPoint (pts1[3].x, pts1[3].y), 10, Scalar(255,255,255), 2, 8, 0);
                 */
                pts2.clear ();
                pts2.push_back(Point2f(0,0));
                pts2.push_back(Point2f(qrsize,0));
                pts2.push_back(Point2f(0, qrsize));
                pts2.push_back(Point2f(qrsize, qrsize));
                // Do perspective transform
                Mat M = getPerspectiveTransform (pts1,pts2);
                warpPerspective (rawFrame, raw, M, Size (qr.cols,qr.rows));
                copyMakeBorder (raw, qr, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(255,255,255));

                flag = 1;
                return ;
            }
    flag = 0;
}

bool ckRatio (int len1, int len2, int len3, int len4, int len5) {
    if (abs (len1 - len2) > len1 * pickRatioL) return 0;
    if (abs (len1 * 3 - len3) > len1 * pickRatioL * 3) return 0;
    if (abs (len1 - len4) > len1 * pickRatioL) return 0;
    if (abs (len1 - len5) > len1 * pickRatioL) return 0;
    return 1;
}

bool ckArea (int area1, int area2, int area3) {
    if (abs (area1 * 2 - area2 * 3) > area1 * 2 * pickRatioA) return 0;
    if (abs (area1 * 3 - area3 * 8) > area1 * 3 * pickRatioA) return 0;
    return 1;
}

void fillarea (int x, int y, int cnt) {
    if (v[x][y]) return;
    q.clear (); 
    q.push_back (Point (x, y));
    for (Point nowX; q.size (); q.pop_front ()) {
        nowX = q[0];
        candidates[cnt].push_back (Point (nowX.x, nowX.y));
        for (int i = 0; i < 8; ++i) {
            Point to = nowX + go[i];
            if (~to.x && ~to.y && to.x < frame.cols && to.y < frame.rows && 
                    bin.at<uchar>(to.y, to.x) == bin.at<uchar>(y, x) && !v[to.x][to.y]) {
                v[to.x][to.y] = 1;
                q.push_back (to);
            }
        }
    }
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
        if (ckArea (area[f[x][y]], area[f[x][y2]], area[f[x][y3]])) {
            vector<Point> tmp; tmp.clear ();
            candidates.push_back (tmp);
            int cnt = candidates.size () - 1;
            fillarea (x, y, cnt);
            fillarea (x, y2, cnt);
            fillarea (x, y3, cnt);
            fillarea (x, y4, cnt);
            fillarea (x, y5, cnt);
            if (candidates[cnt].size() == 0) candidates.resize (cnt);
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
            vector<Point> tmp;
            candidates.push_back (tmp);
            int cnt = candidates.size () - 1;
            fillarea (x, y, cnt);
            fillarea (x2, y, cnt);
            fillarea (x3, y, cnt);
            fillarea (x4, y, cnt);
            fillarea (x5, y, cnt);
            if (candidates[cnt].size() == 0) candidates.resize (cnt);
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
    area[cnt] = 1;
    for (Point nowX; q.size (); q.pop_front ()) {
        nowX = q[0];
        for (int i = 0; i < 8; ++i) {
            Point to = nowX + go[i];
            if (~to.x && ~to.y && to.x < frame.cols && to.y < frame.rows && 
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

void findHull (vector<Point> points, vector<Point> &res) {
    res.clear ();
    Point nowX (maxn, maxn);
    for (int i = 0; i < points.size (); ++i) {
        if (nowX.x > points[i].x) nowX = points[i];
        for (int j = 0; j < 8; ++j) {
            Point to = points[i] + go[j];
            if (~to.x && ~to.y && to.x < frame.cols && to.y < frame.rows && !v[to.x][to.y])  {
                vhull[points[i].x][points[i].y] = 1;
                break;
            }
        }
    }
    for (Point to; ; nowX = to) {
        vhull[nowX.x][nowX.y] = 0;
        res.push_back (nowX);
        bool flag = 1;
        for (int i = 0; i < 8; ++i) {
            to = nowX + go[i];
            if (~to.x && ~to.y && to.x < frame.cols && to.y < frame.rows && vhull[to.x][to.y]) {
                flag = 0;
                break;
            }
        }
        if (flag) break;
    }
    //printf ("%d\n", res.size ());
}

int parameter_init (int argc, const char *argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "show this message.")
        ("image", "use image detection.")
        ("size", po::value<int>(), "set up the qr code size, default 100.")
        ("athre", po::value<double>(), "set up the thresold of area constraint, default 0.2.")
        ("dthre", po::value<double>(), "set up the thresold of distance constraint, default 0.005.")
        ("lratio", po::value<double>(), "set up the ratio of line proportion constraint, default 0.5.")
        ("aratio", po::value<double>(), "set up the ratio of area proportion constraint, default 0.5.")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        cout << desc << endl;
        return EOF;
    }
    if (vm.count ("image")) {
        cout << "Image detection." << endl;
        useimage = 1;
    }
    if (vm.count ("size")) {
        qrsize = vm["size"].as<int> ();
        printf ("QRcode size is set as %d\n", qrsize);
    }
    if (vm.count ("athre")) {
        areathre = vm["athre"].as<double> ();
        printf ("area constraint thresold is set as %lf\n", areathre);
    }
    if (vm.count ("dthre")) {
        distthre = vm["dthre"].as<double> ();
        printf ("distance constraint thresold is set as %lf\n", distthre);
    }
    if (vm.count ("lratio")) {
        pickRatioL = vm["lratio"].as<double> ();
        printf ("line proportion constraint thresold is set as %lf\n", pickRatioL);
    }
    if (vm.count ("aratio")) {
        pickRatioA = vm["aratio"].as<double> ();
        printf ("area proportion constraint thresold is set as %lf\n", pickRatioA);
    }
    return 0;
}

int main(int argc, const char *argv[]) {
    
    // Set up parameters
    if (parameter_init (argc, argv)) return 0;

    VideoCapture capture(0);

    Mat gray(frame.size(), CV_MAKETYPE(frame.depth(), 1));
    Mat marked(frame.size(), CV_MAKETYPE(frame.depth(), 1));
    Mat detected_edges(frame.size(), CV_MAKETYPE(frame.depth(), 1));
    Mat qr;
    vector<Point> raw;

    capture >> frame;
//    frame = imread ("../data/4.jpg");

    cout << "Press any key to return." << endl;

    for (int key = -1; !~key; capture >> frame) {
        candidates.clear (); FIP.clear ();
        memset (v, 0, sizeof (v));
        memset (vhull, 0, sizeof (vhull));

        frame.copyTo (rawFrame);
        // Change to grayscale
        cvtColor (rawFrame, gray, CV_RGB2GRAY);
        threshold (gray, bin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

        // Find all continuous blocks
        floodfill ();

        // Find FIP candidates
        MarkLine ();

        FIP.resize (candidates.size ());
        for (int i = 0; i < candidates.size (); ++i ) {
            findHull (Mat(candidates[i]), raw);
            approxPolyDP (Mat(raw), FIP[i], arcLength (Mat(raw), true) * 0.02, true);
            for (int j = 0; j < FIP[i].size (); ++j) 
                printc (FIP[i][j].x, FIP[i][j].y, 0);
        }

        bool flag;
        findQR (qr, flag);
        if (flag) imshow ("QR", qr);
        imshow ("Image", frame);
        imshow ("Bin", bin);

        key = waitKey (100);
    }

    return 0;
}
