#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
struct keyframes{
    Mat descriptors;
    Mat image;
    vector<Point2d> keypoints;
    Mat pose;
};
struct submap{
    map<Mat,Point3d> landmarks;
    vector<keyframes> keys;
    vector<keyframes> frames;
};
static void computeFundamentalMatrix(const vector<KeyPoint> positions1,const vector<KeyPoint> positions2,
                                    vector<Point2d>& inlierPositions1,
                                     vector<Point2d>& inlierPositions2,
                                    vector<DMatch>& matches,Mat& F)
{
    vector<unsigned char> status;
    // construct aligned position arrays
    std::vector< DMatch > matches1;
    vector<Point2d> inputs1;
    vector<Point2d> inputs2;
    for (int i=0; i<matches.size(); i++) {
        inputs1.push_back(positions1[matches[i].queryIdx].pt);
        inputs2.push_back(positions2[matches[i].trainIdx].pt);
        matches1.push_back(matches[i]);
    }
    // fundamental matrix estimation using eight point algorithm with RANSAC
    F = findFundamentalMat(inputs1, inputs2, CV_FM_RANSAC,0.1,0.99, status);
    // construct aligned inlier position arrays
    inlierPositions1.clear();
    inlierPositions2.clear();
    vector< DMatch > matches2;
    for(int i = 0; i < status.size(); i++) {
        if (status[i]) {
            inlierPositions1.push_back(inputs1[i]);
            inlierPositions2.push_back(inputs2[i]);
            matches2.push_back(matches1[i]);
        }
    }
    // use the inliers and compute F again
    vector<Point2d> newInputs1;
    vector<Point2d> newInputs2;
    //vector<DMatch> selectedMatches;
    for (int i=0; i<status.size(); i++) {
        if (status[i]) {
            newInputs1.push_back(inputs1[i]);
            newInputs2.push_back(inputs2[i]);
        }
    }
    F = findFundamentalMat(newInputs1, newInputs2, CV_FM_8POINT);
    matches = matches2;
}
static bool ExtractRTfromE(const Mat& E,
                           Mat& R1, Mat& R2, 
                           Mat& t1, Mat& t2)
{
    //Using svd decomposition
    Mat svd_u, svd_vt, svd_w;
    SVD::compute(E,svd_u,svd_vt,svd_w);
    // compute the two possible R and t given the E
    double data[] = {0.0 , -1.0, 0.0, 
                1.0, 0.0, 0.0,
                0.0, 0.0, 1.0};
    Mat W(3, 3, CV_64F, data);
    double data1[] = {0.0 , 1.0, 0.0, 
                -1.0, 0.0, 0.0,
                0.0, 0.0, 1.0};
    Mat Wt(3, 3, CV_64F, data1);
    R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
    R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
    t1 = svd_u.col(2); //u3
    t2 = -svd_u.col(2); //u3
    return true;
}
static void constructRt(const Mat& R, const Mat& t, Mat& Rt)
{
    Rt.create(3,4,CV_64F);
    R.copyTo(Rt(Rect(0,0,3,3)));
    Rt.at<double>(0,3) = t.at<double>(0);
    Rt.at<double>(1,3) = t.at<double>(1);
    Rt.at<double>(2,3) = t.at<double>(2);
}
void solveHLS(const Mat& A, Mat& x) {
    SVD svd(A, SVD::MODIFY_A);
    Mat vt = svd.vt;
    x = vt.row(vt.rows-1);
    return;
}

void dehomogenize(Mat& X)
{
    X/=X.at<double>(3);
}

void Mat2Point3d(const Mat& mat, Point3d& point3d)
{
    point3d.x = mat.at<double>(0);
    point3d.y = mat.at<double>(1);
    point3d.z = mat.at<double>(2);
}

static bool TriangulateSinglePointFromTwoView(const Point2d& pts1, const Point2d& pts2,  
                                              const Mat& Rt1, const Mat& Rt2, const Mat& K1, const Mat& K2,
                                              Point3d& result, bool countFront = false)
{
    // compute camera matrix
    Mat P1 = K1 * Rt1; Mat P2 = K2 * Rt2;

    // build the linear system
    Mat A;
    A.create(4, 4, CV_64F);
    A.at<double>(0,0) = P1.at<double>(0,0)-P1.at<double>(2,0)*pts1.x;
    A.at<double>(0,1) = P1.at<double>(0,1)-P1.at<double>(2,1)*pts1.x;
    A.at<double>(0,2) = P1.at<double>(0,2)-P1.at<double>(2,2)*pts1.x;
    A.at<double>(0,3) = P1.at<double>(0,3)-P1.at<double>(2,3)*pts1.x;
    A.at<double>(1,0) = P1.at<double>(1,0)-P1.at<double>(2,0)*pts1.y;
    A.at<double>(1,1) = P1.at<double>(1,1)-P1.at<double>(2,1)*pts1.y;
    A.at<double>(1,2) = P1.at<double>(1,2)-P1.at<double>(2,2)*pts1.y;
    A.at<double>(1,3) = P1.at<double>(1,3)-P1.at<double>(2,3)*pts1.y;
    A.at<double>(2,0) = P2.at<double>(0,0)-P2.at<double>(2,0)*pts2.x;
    A.at<double>(2,1) = P2.at<double>(0,1)-P2.at<double>(2,1)*pts2.x;
    A.at<double>(2,2) = P2.at<double>(0,2)-P2.at<double>(2,2)*pts2.x;
    A.at<double>(2,3) = P2.at<double>(0,3)-P2.at<double>(2,3)*pts2.x;
    A.at<double>(3,0) = P2.at<double>(1,0)-P2.at<double>(2,0)*pts2.y;
    A.at<double>(3,1) = P2.at<double>(1,1)-P2.at<double>(2,1)*pts2.y;
    A.at<double>(3,2) = P2.at<double>(1,2)-P2.at<double>(2,2)*pts2.y;
    A.at<double>(3,3) = P2.at<double>(1,3)-P2.at<double>(2,3)*pts2.y;
    
    // solve it
    Mat X; solveHLS(A, X);
    dehomogenize(X);
    Mat2Point3d(X, result);

    // check whether it is in front of the both camera
    if (countFront) {
        Mat pc1 = Rt1*X.t();
        Mat pc2 = Rt2*X.t();
        if (pc1.at<double>(2)>0 && pc2.at<double>(2)>0) {
            return true;
        }
        else return false;
    }
    return true;
}



vector<int> TriangulateMultiplePointsFromTwoView(const vector<Point2d>& pts1, const vector<Point2d>& pts2, 
                                                 const Mat& Rt1, const Mat& Rt2, const Mat& K1, const Mat& K2,
                                                 vector<Point3d>& result, bool countFront = false)
{
    int count = 0;
    vector<int> points;
    result.clear();
    for (int i=0; i<pts1.size(); i++)
    {
        Point3d point3d;
        bool front = TriangulateSinglePointFromTwoView(pts1[i], pts2[i], Rt1, Rt2, K1, K2, point3d, countFront);
        if (front) count++;
        result.push_back(point3d);
        points.push_back(i);
    }
    return points;
}


submap initialise(Mat src , Mat dst){
    submap sub;
    keyframes frame1,frame2;
    frame1.image = src;
    frame2.image = dst;
    Ptr<SIFT> extractor = SIFT::create();
    vector<KeyPoint> keypoints_1,keypoints_2;
    Mat descriptor1,descriptor2;
    extractor->detect(src, keypoints_1);
    extractor->compute(src, keypoints_1, descriptor1);
    extractor->detect(src, keypoints_2);
    extractor->compute(src, keypoints_2, descriptor2);
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptor1, descriptor2, matches );
    Mat F;
    vector<Point2d> goodPositions1,goodPositions2;
    computeFundamentalMatrix(keypoints_1,keypoints_2,goodPositions1,goodPositions2,matches,F);
    double data[] = {1189.46 , 0.0, 805.49, 
                0.0, 1191.78, 597.44,
                0.0, 0.0, 1.0};//Camera Matrix
    Mat K(3, 3, CV_64F, data);
    cv::Mat K_tp = cv::Mat(K.cols, K.rows, CV_32F);
    cv::Mat Kinv= K.inv();
    transpose(Kinv,K_tp);
    Mat E;
    E= K_tp*F*K;
    Mat R1, R2, T1, T2;
    ExtractRTfromE(E, R1, R2, T1, T2);
    Mat Rts[4];
    constructRt(R1, T1, Rts[0]); constructRt(R1, T2, Rts[1]);
    constructRt(R2, T1, Rts[2]); constructRt(R2, T2, Rts[3]);
    Mat I = Mat::eye(3,4, CV_64F);
    frame1.pose=I;
    int bestRtIndex = -1;
    int maxCount =-1;
    vector<int> points;
    vector<Point3d> result;
    for (int i=0; i < 4; i++) {
        vector<Point3d> tmpResult;
        Mat Rt = Rts[i];
        vector<int> temp = TriangulateMultiplePointsFromTwoView(goodPositions1, goodPositions2, I, Rt, K, K, tmpResult, true);
        if (maxCount < temp.size()) {
            maxCount = temp.size();
            bestRtIndex = i;
            points = temp;
            result = tmpResult;
            frame2.pose=Rt;
        }
    }
    vector< vector<double> > vdescriptors;
    vdescriptors.resize(points.size());
    vector< vector<double> > vdescriptors1;
    vdescriptors1.resize(points.size());
    for(int i=0;i<points.size();i++){
        frame1.keypoints.push_back(goodPositions1[points[i]]);
        frame2.keypoints.push_back(goodPositions2[points[i]]);
        int srci,dsti;
        srci=matches[points[i]].queryIdx;
        dsti=matches[points[i]].trainIdx;
        for(unsigned long j=0;j<descriptor1.cols;j++){
            vdescriptors[i].push_back(descriptor1[srci][j]);
        }
        for(unsigned long j=0;j<descriptor2.cols;j++){
            vdescriptors1[i].push_back(descriptor2[dsti][j]);
        }
    }
    cv::Mat acdescriptors1(vdescriptors.size(), vdescriptors[0].size(), CV_64FC1);
    cv::Mat acdescriptors2(vdescriptors1.size(), vdescriptors1[0].size(), CV_64FC1);
    for(int i=0;i<vdescriptors.size();i++){
        for (int j = 0; j < vdescriptors[i].size(); ++j)
        {
            acdescriptors1.at<double>(i,j)=vdescriptors[i][j];
        }
    }
    for(int i=0;i<vdescriptors1.size();i++){
        for (int j = 0; j < vdescriptors1[i].size(); ++j)
        {
            acdescriptors2.at<double>(i,j)=vdescriptors1[i][j];
        }
    }
    frame1.descriptors=acdescriptors1;
    frame2.descriptors=acdescriptors2;
    sub.frames.push_back(frame1);
    sub.frames.push_back(frame2);
    sub.landmarks[frame2.descriptors]=result;
    return sub;
}
bool addtoMap(Mat Curframe,submap sub){
    keyframes prevframe,curframe;
    prevframe = sub.frames[sub.frames.size()-1];
    vector<Point3d> map3d;
    map3d = sub.landmarks[prevframe.descriptors];
    cv::Rect r( 0, 0, 3, 3);
    cv::Mat A = sub.frames.pose;
    cv::Mat Rvec = A(r).clone();
    cv::Mat tvec;
    tvec.at<double>(0) = A.at<double>(0,3);
    tvec.at<double>(1) = A.at<double>(1,3);
    tvec.at<double>(2) = A.at<double>(2,3);
    double data[] = {1189.46 , 0.0, 805.49, 
                0.0, 1191.78, 597.44,
                0.0, 0.0, 1.0};//Camera Matrix
    Mat K(3, 3, CV_64F, data);
    cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);   // Distortion vector
    distCoeffs.at<double>(0) = -7.9134632415085826e-001;
    distCoeffs.at<double>(1) = 1.5623584435644169e+000;
    distCoeffs.at<double>(2) = -3.3916502741726508e-002;
    distCoeffs.at<double>(3) = -1.3921577146136694e-002;
    distCoeffs.at<double>(4) = 1.1430734623697941e+002;
    vector<Point2d> imagePoints;
    cv::projectPoints(map3d, RVec, tVec,K, distCoeffs, imagePoints);
    vector<Keypoint> keypoints_1,keypoints_2;
    keypoints_1 = prevframe.keypoints;
    vector<Point3d> impLandmarks;
    cv::Rect rect(cv::Point(), Curframe.size());
    for(unsigned long i=0;i<imagePoints.size();i++){
        if (rect.contains(imagePoints[i])){
            keypoints_2.push_back(Keypoint(imagePoints[i],5));
            impLandmarks.push_back(map3d[i]);
        }
    }
    if(keypoints_2.size()<10)
        return false;
    vector<Point3d> vimplandmarks;
    Mat descriptor;
    extractor->compute(src, keypoints, descriptor);
    FlannBasedMatcher matcher;
    std::vector< DMatch > rawmatches;
    matcher.match( descriptor, prevframe.descriptors, matches);
    vector<Point2d> tmp0, tmp1;
    Mat f;
    computeFundamentalMatrix(keypoints_2, keypoints_1, rawMatches, tmp0, tmp1, f);
    for (int i=0; i<rawMatches.size(); i++) {
        vimplandmarks.push_back(impLandmarks[rawMatches[i].queryIdx]);
        curframe.keypoints.push_back(keypoints_2[rawMatches[i].queryIdx]);
        int srci,dsti;
        srci=rawMatches[i].queryIdx;
        for(unsigned long j=0;j<descriptors1.cols;j++){
            vdescriptors[i].push_back(descriptors1[srci][j]);
        }
    }
    cv::Mat acdescriptors1(vdescriptors.size(), vdescriptors[0].size(), CV_64FC1);
    for(int i=0;i<vdescriptors.size();i++){
        for (int j = 0; j < vdescriptors[i].size(); ++j)
        {
            acdescriptors1.at<double>(i,j)=vdescriptors[i][j];
        }
    }
    Mat rvec, tvec, R, Rt;
    solvePnPRansac(vimplandmarks, tmp0, K, distCoeff, rvec, tvec);
    Rodrigues(rvec, R); constructRt(R, tvec, Rt);
    curframe.pose=Rt;
    curframe.descriptor=acdescriptors1;
    curframe.image=Curframe;
    sub.frames.push_back(curframe);
    return true;
}
int main(){
    VideoCapture cap(0);
    int count = 0;
    int keycount =0;
    vector<submap> submaps;
    while(1){
        Mat src;
        cap>>src;
        if(!src.data){
            cout<<"Error";
            return 0;
        }
        if(count==0){
            keycount = 0;
            Mat dst;
            cap>>dst;
            submap sub = initialise(src,dst);
            count = 2;
            sub.keys.push_back(0);
            sub.keys.push_back(1);
            submaps.push_back(sub);
            continue;
        }
        else if(count == thresh){
            count =0;
            continue;
        }
        else{
            submap sub = submaps[submaps.size()-1];
            if(!addtoMap(src,sub));
        }
    }
    return 0;
}

