/**
 * @file lsd_opencv_example.cpp
 *
 * Test the LSD algorithm with OpenCV
 */
//#include <opencv2/highgui/highgui.h>
#include <cv.h>
#include <lsd.h>
#include <opencv2/core/core.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <string>

//#include <mlpack/methods/kmeans/kmeans.hpp>
//using namespace mlpack::kmeans;
//extern arma::mat data;
//extern size_t clusters;
//arma::Row<size_t> assignments;
//arma::mat centroids;

#define USE_PPHT
#define MAX_NUM_LINES	200


#include "MSAC.h"
#include "selfdef.h"
#include "lm.h"
#include <ceres/ceres.h>
#include <Eigen/Core>
#include "k-means.h"
#include "algorithm"
//#include <cstdlib>
//#include "queue"
//using ceres::AutoDiffCostFunction;
using namespace cv;
using namespace std;
using namespace ceres;

typedef Eigen::Matrix<double,2,1> OnLinePoint;
typedef std::vector<OnLinePoint> Lineready;
Lineready LinePoints;
double pi=3.14159265358979323846264;
int flag=1;
int flag2=0;
int flagcontinue=0;
double theta = -90;
std::vector<cv::Mat> VPpre1;
std::vector<double> linemark;
extern int frame;

cv::Point2f VPpre2;

std::vector<cv::Point2f> VPvector;
//std::queue<cv::Point2f> VPqueue;

cv::VideoWriter writer1("homographyvideo.avi",0,20.0,Size(1200,1080),true);

struct HomographyCostFunctor {
    HomographyCostFunctor(OnLinePoint line1p1, OnLinePoint line1p2, OnLinePoint line2p1, OnLinePoint line2p2) : _line1p1(line1p1), _line1p2(line1p2), _line2p1(line2p1), _line2p2(line2p2) {}

    template <typename T> bool operator()(const T* const theta, T* residual) const {//theta 是优化变量 T强制类型转换指针

        //residual[0]=T(cos(theta[0]/T(180)*T(pi)));
        T thet=theta[0]/T(180)*T(pi);
        T sintheta = T(ceres::sin(thet));
        T costheta = T(ceres::cos(thet));
        T x1=T(_line1p1(0));
        T y1=T(_line1p1(1));
        T x2=T(_line1p2(0));
        T y2=T(_line1p2(1));
        T x3=T(_line2p1(0));
        T y3=T(_line2p1(1));
        T x4=T(_line2p2(0));
        T y4=T(_line2p2(1));

        //Eigen::Matrix<double,3,3> K,Ry,H;
        //T K << 1.0074*pow(10,3),0,6.5638*pow(10,2),0,9.9883*pow(10,2),3.3974*pow(10,2),0,0,1;

        //Ry << ceres::cos(theta/180*pi),0,ceres::sin(theta/180*pi),0,1,0,ceres::sin(-theta/180*pi),0,ceres::cos(theta/180*pi);
        //H=K*Ry*K.inverse();

        T tan1=((339.74*costheta + 221.36047369465951955529084772682*sintheta + 1.0*y1 - 0.33724439150287869763748262854874*sintheta*x1 - 339.74)/(1.0*costheta + 0.65155846734167162993845542981934*sintheta - 0.00099265435775263053404804447091523*sintheta*x1) - (339.74*costheta + 221.36047369465951955529084772682*sintheta + 1.0*y2 - 0.33724439150287869763748262854874*sintheta*x2 - 339.74)/(1.0*costheta + 0.65155846734167162993845542981934*sintheta - 0.00099265435775263053404804447091523*sintheta*x2))/((1435.0699467937264244590033750248*sintheta + x1*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(1.0*costheta + 0.65155846734167162993845542981934*sintheta - 0.00099265435775263053404804447091523*sintheta*x1) - (1435.0699467937264244590033750248*sintheta + x2*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(1.0*costheta + 0.65155846734167162993845542981934*sintheta - 0.00099265435775263053404804447091523*sintheta*x2));
        T tan2=((339.74*costheta + 221.36047369465951955529084772682*sintheta + 1.0*y3 - 0.33724439150287869763748262854874*sintheta*x3 - 339.74)/(1.0*costheta + 0.65155846734167162993845542981934*sintheta - 0.00099265435775263053404804447091523*sintheta*x3) - (339.74*costheta + 221.36047369465951955529084772682*sintheta + 1.0*y4 - 0.33724439150287869763748262854874*sintheta*x4 - 339.74)/(1.0*costheta + 0.65155846734167162993845542981934*sintheta - 0.00099265435775263053404804447091523*sintheta*x4))/((1435.0699467937264244590033750248*sintheta + x3*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(1.0*costheta + 0.65155846734167162993845542981934*sintheta - 0.00099265435775263053404804447091523*sintheta*x3) - (1435.0699467937264244590033750248*sintheta + x4*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(1.0*costheta + 0.65155846734167162993845542981934*sintheta - 0.00099265435775263053404804447091523*sintheta*x4));


        T angle=tan1*tan1+tan2*tan2;
        T parallel=- (1014854.76*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)))/(sintheta*sintheta*(x3 - x4)*(x3 - x4)*((1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta)*(sintheta) + (1014854.76*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)))/((x3 - x4)*(x3 - x4)*(sintheta)*(sintheta)) + 1.0)) - (1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta*sintheta*((1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta)*(sintheta) + (1014854.76*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2))*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2)))/ceres::sqrt(((x1 - x2)*(x1 - x2)*(sintheta)*(sintheta)) + 1.0))*ceres::sqrt(((1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta)*(sintheta) + (1014854.76*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)))/((x3 - x4)*(x3 - x4)*(sintheta)*(sintheta)) + 1.0)));

        residual[0]=angle;

        //residual[0]=tan1*tan1+tan2*tan2;


        //residual[0] =- (1014854.76*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)))/(sintheta*sintheta*(x3 - x4)*(x3 - x4)*((1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta)*(sintheta) + (1014854.76*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)))/((x3 - x4)*(x3 - x4)*(sintheta)*(sintheta)) + 1.0)) - (1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta*sintheta*((1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta)*(sintheta) + (1014854.76*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2))*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2)))/ceres::sqrt(((x1 - x2)*(x1 - x2)*(sintheta)*(sintheta)) + 1.0))*ceres::sqrt(((1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta)*(sintheta) + (1014854.76*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)))/((x3 - x4)*(x3 - x4)*(sintheta)*(sintheta)) + 1.0)));
//
//
//  residual[0] = - (1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))/((1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)) + 0.00000098536267396528740723450910354896*(sintheta*(x3 - x4))*(sintheta*(x3 - x4)) + 1.0*((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))*((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))) - (1007.4*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))/(sintheta*((1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)) + 0.00000098536267396528740723450910354896*(sintheta*(x3 - x4))*(sintheta*(x3 - x4)) + 1.0*((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4)))*ceres::sqrt((1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta)*(sintheta) + (1014854.76*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2))*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2)))/((x1 - x2)*(x1 - x2)*(sintheta)*(sintheta)) + 1.0));
		//residual[0] = - (1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))/((1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)) + 0.00000098536267396528740723450910354896*(sintheta*(x3 - x4))*(sintheta*(x3 - x4)) + 1.0*((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))*((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))) - (1007.4*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))/(sintheta*((1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)) + 0.00000098536267396528740723450910354896*(sintheta*(x3 - x4))*(sintheta*(x3 - x4)) + 1.0*abs((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4)))*sqrt((1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta)*(sintheta) + (1014854.76*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2))*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2)))/((x1 - x2)*(x1 - x2)*(sintheta)*(sintheta)) + 1.0));
        return true;
    }

private:
    OnLinePoint _line1p1;
    OnLinePoint _line1p2;
    OnLinePoint _line2p1;
    OnLinePoint _line2p2;
};




struct HomographyParallel {
    HomographyParallel(OnLinePoint line1p1, OnLinePoint line1p2, OnLinePoint line2p1, OnLinePoint line2p2) : _line1p1(line1p1), _line1p2(line1p2), _line2p1(line2p1), _line2p2(line2p2) {}

    template <typename T> bool operator()(const T* const tantheta1, const T* const tantheta2, T* residual) const {//theta 是优化变量 T强制类型转换指针

        //residual[0]=T(cos(theta[0]/T(180)*T(pi)));
        //T thet=theta[0]/T(180)*T(pi);
        //T sintheta = T(ceres::sin(thet));
        //T costheta = T(ceres::cos(thet));
        T x1=T(_line1p1(0));
        T y1=T(_line1p1(1));
        T x2=T(_line1p2(0));
        T y2=T(_line1p2(1));
        T x3=T(_line2p1(0));
        T y3=T(_line2p1(1));
        T x4=T(_line2p2(0));
        T y4=T(_line2p2(1));

        residual[0] = pow(tantheta1,2)+pow(tantheta2,2);
        //residual[0] = - (1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))/((1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)) + 0.00000098536267396528740723450910354896*(sintheta*(x3 - x4))*(sintheta*(x3 - x4)) + 1.0*((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))*((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))) - (1007.4*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))/(sintheta*((1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)) + 0.00000098536267396528740723450910354896*(sintheta*(x3 - x4))*(sintheta*(x3 - x4)) + 1.0*((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4)))*ceres::sqrt((1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta)*(sintheta) + (1014854.76*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2))*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2)))/((x1 - x2)*(x1 - x2)*(sintheta)*(sintheta)) + 1.0));
        //residual[0] = - (1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))/((1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)) + 0.00000098536267396528740723450910354896*(sintheta*(x3 - x4))*(sintheta*(x3 - x4)) + 1.0*((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))*((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))) - (1007.4*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4))/(sintheta*((1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4))*(1.0*y4 - 1.0*y3 + 0.33724439150287869763748262854874*sintheta*(x3 - x4)) + 0.00000098536267396528740723450910354896*(sintheta*(x3 - x4))*(sintheta*(x3 - x4)) + 1.0*abs((1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(x3 - x4)))*sqrt((1014854.76*(1.0*costheta - 0.65155846734167162993845542981934*sintheta)*(1.0*costheta - 0.65155846734167162993845542981934*sintheta))/(sintheta)*(sintheta) + (1014854.76*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2))*(1.0*y2 - 1.0*y1 + 0.33724439150287869763748262854874*sintheta*(x1 - x2)))/((x1 - x2)*(x1 - x2)*(sintheta)*(sintheta)) + 1.0));
        return true;
    }

private:
    OnLinePoint _line1p1;
    OnLinePoint _line1p2;
    OnLinePoint _line2p1;
    OnLinePoint _line2p2;
};




void processImage(MSAC &msac, int numVps, cv::Mat &imgGRAY, cv::Mat &outputImg, cv::Mat &srctemp, cv::Point2f &VPET)
{std::cout<<"!1"<<std::endl;
	cv::Mat imgCanny;

	// Canny
	cv::Canny(imgGRAY, imgCanny, 180, 120, 3);
       /* imshow("imgCanny",imgCanny);*/
	// Hough
	vector<vector<cv::Point> > lineSegments;
	vector<cv::Point> aux;
#ifndef USE_PPHT
	vector<Vec2f> lines;
	cv::HoughLines( imgCanny, lines, 1, CV_PI/180, 200);
        //cv::HoughLines( imgCanny, lines, 1, CV_PI/180, 350);
	for(size_t i=0; i< lines.size(); i++)
	{
		float rho = lines[i][0];
		float theta = lines[i][1];

		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;

		Point pt1, pt2;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));

		aux.clear();
		aux.push_back(pt1);
		aux.push_back(pt2);
		lineSegments.push_back(aux);

		line(outputImg, pt1, pt2, CV_RGB(0, 0, 0), 1, 8);
	
	}
	//imshow("HOUGH LINES",outputImg);
	waitKey(0);
	
#else
	vector<Vec4i> lines;	
	int houghThreshold = 70;
	if(imgGRAY.cols*imgGRAY.rows < 400*400)
		houghThreshold = 100;		
	
	cv::HoughLinesP(imgCanny, lines, 1, CV_PI/180, houghThreshold, 10,10);

	while(lines.size() > MAX_NUM_LINES)
	{
		lines.clear();
		houghThreshold += 10;
		cv::HoughLinesP(imgCanny, lines, 1, CV_PI/180, houghThreshold, 10, 10);
	}
int width,height;
width=imgGRAY.cols;
height=imgGRAY.rows;
/////////////////////////////////////////////////////////////////////////////调整roi和线的长度 这句话决定了是否要进行直线滤波
FilterHorizonLine(lines,width,height);

Mat srctemp1=srctemp.clone();
	for(size_t i=0; i<lines.size(); i++)
	{		
		Point pt1, pt2;
		pt1.x = lines[i][0];
		pt1.y = lines[i][1];
		pt2.x = lines[i][2];
		pt2.y = lines[i][3];
		
		//std::cout<<"!";
		
//		line(outputImg, pt1, pt2, CV_RGB(0,0,255), 1);//只画hough lines
	/*	line(srctemp1, pt1, pt2, CV_RGB(0,0,255), 1);//只画hough lines*/
//		imshow("HoughLines",outputImg);
//		double alpha=1;
//		double beta=2;
//		Mat dstimage[3],dstimagedl[3],dstimage1;
//		Mat mv[3],mvdl[3];
//		Mat processed,processeddl;
                //split(src,mvdl);
	//	addWeighted(srctemp,alpha,outputImg,beta,0.0,dstimage1);
		//addWeighted(mv[0],alpha,lsd,beta,0.0,dstimage[0]);
		//addWeighted(mv[1],alpha,lsd,beta,0.0,dstimage[1]);
		//addWeighted(mv[2],alpha,lsd,beta,0.0,dstimage[2]);
    
		//addWeighted(mvdl[0],alpha,lsddeeplearning,beta,0.0,dstimagedl[0]);
		//addWeighted(mvdl[1],alpha,lsddeeplearning,beta,0.0,dstimagedl[1]);
		//addWeighted(mvdl[2],alpha,lsddeeplearning,beta,0.0,dstimagedl[2]);
    
		//merge(dstimage,3,processed);
		//merge(dstimagedl,3,processeddl);
		
//		imwrite("mixdl.jpg",srctemp1);
	//	imshow("mixdl",dstimage1);
	//	imwrite("mixdl.jpg",dstimage1);
		//imshow("mixdl",processeddl);
		//cv::imwrite("mix.jpg",processed);
		//cv::imwrite("mixdl.jpg",processeddl);
		
		
		
		
		//line(outputImgH, pt1, pt2, CV_RGB(0,0,0), 2);
		/*circle(outputImg, pt1, 2, CV_RGB(255,255,255), CV_FILLED);
		circle(outputImg, pt1, 3, CV_RGB(0,0,0),1);
		circle(outputImg, pt2, 2, CV_RGB(255,255,255), CV_FILLED);
		circle(outputImg, pt2, 3, CV_RGB(0,0,0),1);*/

		// Store into vector of pairs of Points for msac
		aux.clear();
		aux.push_back(pt1);
		aux.push_back(pt2);
		lineSegments.push_back(aux);
	}
	/*imshow("PHOUGH LINES",srctemp1);*/
#endif

	// Multiple vanishing points
	std::vector<cv::Mat> vps;			// vector of vps: vps[vpNum], with vpNum=0...numDetectedVps
	std::vector<std::vector<int> > CS;	// index of Consensus Set for all vps: CS[vpNum] is a vector containing indexes of lineSegments belonging to Consensus Set of vp numVp
	std::vector<int> numInliers;

	std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;



	// Call msac function for multiple vanishing point estimation
	msac.multipleVPEstimation(lineSegments, lineSegmentsClusters, numInliers, vps, numVps);



	for(int v=0; v<vps.size(); v++)
	{
        std::cout<<"msac vp"<<std::endl;
		printf("VPnum=%d (%.3f, %.3f, %.3f)", v, vps[v].at<float>(0,0), vps[v].at<float>(1,0), vps[v].at<float>(2,0));
		fflush(stdout);
		double vpNorm = cv::norm(vps[v]);
		if(fabs(vpNorm - 1) < 0.001)
		{
			printf("(INFINITE)");
			fflush(stdout);
		}
		printf("\n");
	}
    cout<<"!2"<<endl;


    std::cout<<"vps num size = "<<vps.size()<<std::endl;


    // Draw line segments according to their cluster
	//msac.drawCS(outputImg, lineSegmentsClusters, vps);
//	et.x=vps[0].at<float>(0,0);
//	et.y=vps[0].at<float>(1,0);

    //std::vector<Point2f> VPvectortemp;
   // std::queue<cv::Point2f> queuetemp;
    if(VPvector.size()>=31)
   //     if(VPvector.size()>=41)
    {
        //for(std::vector<Point2f>::iterator iter=VPvector.begin()+1;iter!=VPvector.end();iter++)
        //{
         //VPvectortemp.push_back(*iter);
        //}
        //VPvector.clear();
        //VPvector=VPvectortemp;
        //VPvector.begin()=VPvector.begin()+1;
        //VPqueue.pop();
        VPvector.erase(VPvector.begin());



        //cout<<"VPvector.begin() = "<<*(VPvector.begin())<<endl;
        cout<<"VPvector.size() = "<<VPvector.size()<<endl;
        //cout<<"VPvector ="<<VPvector<<endl;
    }
    //VPvector=VPqueue;
//    if(VPvector.size()>=5) {



//        RotatedRect VPEllipse=fitEllipse(VPvector);
//        ellipse(srctemp, VPEllipse, CV_RGB(0, 0, 255));
//        Rect bounds=VPEllipse.boundingRect();
//        rectangle(srctemp, bounds, Scalar(255, 0, 0));//蓝色的框


//        line(srctemp, cv::Point(bounds.x+bounds.width/2, bounds.y), cv::Point(bounds.x+bounds.width/2,bounds.y+bounds.height), CV_RGB(0,0,255), 0.5);


//        for(std::vector<cv::Point2f>::iterator iter=VPvector.begin();iter!=VPvector.end();iter++)
//        {
//            circle(srctemp, *iter, 2, CV_RGB(0, 0, 255), -1);
//        }
        /*circle(srctemp, VPvector[0], 2, CV_RGB(0, 0, 255), -1);
        circle(srctemp, VPvector[1], 2, CV_RGB(0, 0, 255), -1);
        circle(srctemp, VPvector[2], 2, CV_RGB(0, 0, 255), -1);
        circle(srctemp, VPvector[3], 2, CV_RGB(0, 0, 255), -1);
        circle(srctemp, VPvector[4], 2, CV_RGB(0, 0, 255), -1);
        if(VPvector.size()>=6){
           circle(srctemp, VPvector[5], 2, CV_RGB(0, 0, 255), -1);
        }
        if(VPvector.size()>=7){
            circle(srctemp, VPvector[6], 2, CV_RGB(0, 0, 255), -1);
        }
        if(VPvector.size()>=8){
            circle(srctemp, VPvector[7], 2, CV_RGB(0, 0, 255), -1);
        }
        if(VPvector.size()>=9){
            circle(srctemp, VPvector[8], 2, CV_RGB(0, 0, 255), -1);
        }
        if(VPvector.size()>=10){
            circle(srctemp, VPvector[9], 2, CV_RGB(0, 0, 255), -1);
        }*/


//    }


    cv::Point2f etlm;


    if(vps.size()!=0) {



        VPpre1=vps;
        VPET.x=vps[0].at<float>(0,0);
	VPET.y=vps[0].at<float>(1,0);





//        if(VPvector.size()>=5) {
 //           RotatedRect VPEllipse=fitEllipse(VPvector);
  //          ellipse(srctemp, VPEllipse, CV_RGB(0, 0, 255));
   //     }


    cout<<"!3"<<endl;

	
        //std::cout<<"size="<<vps.size()<<std::endl;
	//std::cout<<"vp="<<vps[0].at<float>(0,0)<<"   "<<vps[0].at<float>(1,0)<<"    "<<vps[0].at<float>(2,0)<<std::endl;


        //msac.drawCS(outputImg, lineSegmentsClusters, vps);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////画红线和红点
    //    msac.drawCS(srctemp, lineSegmentsClusters, vps);


        //circle(srctemp, et, 5, CV_RGB(255,0,0));
    }

    cout<<"!3.5"<<endl;

        vector<Vec4i> linesRansac;
        for (unsigned int c = 0; c < lineSegmentsClusters.size(); c++) {
            for (unsigned int i = 0; i < lineSegmentsClusters[c].size(); i++) {
                Point pt1 = lineSegmentsClusters[c][i][0];
                Point pt2 = lineSegmentsClusters[c][i][1];
                Vec4i temp(pt1.x, pt1.y, pt2.x, pt2.y);

                linesRansac.push_back(temp);

                //line(im, pt1, pt2, colors[c], 1);
            }
        }
   //VP为0这些不要 VP wei 0 keyi yong shangyizhan zuowei chushizhi youhua  ransac de xian yao you
    if(vps.size()!=0) {
        etlm = VPET;//VP第一侦要检测出来
        lm(linesRansac, etlm);//yong chu shi zhi fan xiang you hua  huo zhe chong xin you hua

        VPET = etlm;


       // if(VPET.x!=0 && VPET.y!=0 && flag2==0 && VPET.x>0 && VPET.x<1280 && VPET.y>0 && VPET.y<720) { //VP.x!=0 && VP.y!=0 是检测到了   VP.x>0 && VP.x<1280 && VP.y>0 && VP.y<720 只要检测到好点就记录并且更新 消失点就不会再跑出那个小圈子
       //     VPpre2 = VPET;
       //     flag2=1;//不再进去
       // }
       // if(abs(VPET.x-VPpre2.x)>200 || abs(VPET.x-VPpre2.y)>200)//VP没有算出来的话会是（0,0)  LSDVP内部函数会跳过VP计算部分
       // {//突变超过200 认为不好的点 少于200 可以被慢慢带过去
       //     cout<<"!!!!NO VP"<<endl;
       //     VPET=VPpre2;
       // }
       // else
       // {
       //     VPpre2=VPET;
       // }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////优化后的点加入椭圆
        //VPqueue.push(etlm);

        if(VPET.x>0 && VPET.x<1280 && VPET.y>0 && VPET.y<720) {

            VPvector.push_back(VPET);
        }
    }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////画蓝点看不见
    //    circle(srctemp, etlm, 6, CV_RGB(0, 0, 255), 2);
    //    circle(srctemp, etlm, 5, CV_RGB(0, 0, 255), -1);



    // circle(outputImg, etlm, 6, CV_RGB(0, 0, 255), 2);
      //  circle(outputImg, etlm, 5, CV_RGB(0, 0, 255), -1);
      //  double theta = -80;
        ceres::Problem problem;
        for (auto item1:linesRansac) {
        //for (vector<Vec4i>::iterator item1=linesRansac.begin();item1 != linesRansac.end();item1++) {
            for (auto item2:linesRansac) {
            //for (vector<Vec4i>::iterator item2 = item1; item2 != linesRansac.end();item2++) {
                problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<HomographyCostFunctor, 1, 1>(
                                new HomographyCostFunctor(OnLinePoint(item1[0], item1[1]),
                                                          OnLinePoint(item1[2], item1[3]),
                                                          OnLinePoint(item2[0], item2[1]),
                                                          OnLinePoint(item2[2], item2[3]))), NULL,//new CauchyLoss(0.5),
                        &theta);
            }
        }
        Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << "theta=" << theta << std::endl;


        for (auto item:linesRansac) {
          /////////////////////////////////////////////////////////////////////////////////////////////////////////////////画绿线
          //  line(srctemp, etlm, Point2f(0.5 * (item[0] + item[2]), 0.5 * (item[1] + item[3])), CV_RGB(0, 255, 0), 0.5);


            //  line(outputImg, etlm, Point2f(0.5 * (item[0] + item[2]), 0.5 * (item[1] + item[3])), CV_RGB(0, 255, 0),
           //      0.5);
        }

    Eigen::Matrix<double,3,3> K,Ry,H;
    K<<1.0074*pow(10,3),0,6.5638*pow(10,2),0,9.9883*pow(10,2),3.3974*pow(10,2),0,0,1;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////更改角度
  //  if(theta<-95|| theta>-80) {
       //theta=-88;
   // theta=-90;
   // }




    Ry << ceres::cos(theta/180*pi),0,ceres::sin(theta/180*pi),0,1,0,ceres::sin(-theta/180*pi),0,ceres::cos(theta/180*pi);
    H=K*Ry*K.inverse();




    //std::cout<<"H="<<H<<std::endl;
    std::vector<Vec4i> rotationlines;
    //std::vector<vector<double> > clustery;
    std::vector<double> clustery;
    //extern arma::mat data;
    //int z=1;
    cout<<"linesRansac.size()"<<linesRansac.size()<<endl;

    if (linesRansac.size()!=0 && etlm.x>0 && etlm.x<1280 && etlm.y>0 && etlm.y<720) { //有的时候会没有线  就跳过       消失点好的时候才聚类 消失点在图片内认为是好的
        for (auto lines:linesRansac) {
            Eigen::Vector3d begin(lines[0], lines[1], 1), end(lines[2], lines[3], 1), newbegin, newend;
            newbegin = H * begin;
            newend = H * end;
            //std::cout<<"newbegin="<<newbegin<<std::endl;
            //std::cout<<"newend="<<newend<<std::endl;
            newbegin = newbegin / newbegin[2];
            newend = newend / newend[2];
            Vec4i newline(round(newbegin[0]), round(newbegin[1]), round(newend[0]), round(newend[1]));
            rotationlines.push_back(newline);


            clustery.push_back(10000 * (newbegin[1] + newend[1]));
            //data(i)=0.5*(newbegin[1]+newend[1]);
            //z++;
            //std::vector<double> temp;
            //temp.push_back(0.5*(newbegin[1]+newend[1]));
            //temp.push_back(0);
            //clustery.push_back(temp);
        }

//rotation与lineransac对应
        //sort(rotationlines.begin(),rotationlines.end(),[](Vec4i a, Vec4i b){return a[1]+a[3]>b[1]+b[3];});
        //sort(rotationlines.begin(), rotationlines.end(), [](Vec4i a, Vec4i b) { return a[3] > b[3]; });
        //sort(linesRansac.begin(), linesRansac.end(), [](Vec4i a, Vec4i b) { return a[3] > b[3]; });
        sort(rotationlines.begin(), rotationlines.end(), [](Vec4i a, Vec4i b) { return a[3] > b[3]; });
        sort(linesRansac.begin(), linesRansac.end(), [](Vec4i a, Vec4i b) {
            if (a[2] > a[0] && b[2] > b[0]) { return a[3] > b[3]; }
            if (a[2] > a[0] && b[0] > b[2]) { return a[3] > b[1]; }
            if (a[0] > a[2] && b[0] > b[2]) { return a[1] > b[1]; }
            if (a[0] > a[2] && b[2] > b[0]) { return a[1] > b[3]; }
        });


        for (auto lines:rotationlines) {
            cout << "rotation lines" << lines[3] << endl;
        }
        for (auto lines:linesRansac) {
            if (lines[2] >= lines[0]) {
                cout << "Ransac lines" << lines[3] << endl;
            }
            if (lines[2] < lines[0]) {
                cout << "Ransac lines" << lines[1] << endl;
            }
        }
        std::vector<Vec2i> rotationlinesinterval;

//旋转过的线条计算间隔 有纵向拉伸聚类效果不好

        for (int i = 0; i < rotationlines.size() - 1; i++) {
            //Vec2i a(round(ceres::abs(rotationlines[i][1]+rotationlines[i][3]-rotationlines[i+1][1]-rotationlines[i+1][3])),i+1);//利用平行线之间的距离
            int x1 = rotationlines[i][0], y1 = rotationlines[i][1], x2 = rotationlines[i][2], y2 = rotationlines[i][3], x3 = rotationlines[
                    i + 1][0], y3 = rotationlines[i + 1][1], x4 = rotationlines[i + 1][2], y4 = rotationlines[i + 1][3];
//以前的距离是点到直线的距离
//            Vec2i a(round(ceres::abs(
//                    ((y4 - y3) * (x1 + x2) / 2 - (x4 - x3) * (y1 + y2) / 2 + y4 * (x4 - x3) - (y4 - y3) * x4) /
//                    ceres::sqrt(pow((y4 - y3), 2) + pow((x4 - x3), 2)))), i + 1);

            //倾斜的时候无法判断
            //Vec2i a(ceres::abs((y4+y3)*0.5-(y1+y2)*0.5), i + 1);
            Vec2i a(ceres::abs((y4) - (y2)), i + 1);//直线间隔


            //Vec2i a(round(ceres::abs((y1+y2)/2)),i+1);
            //rotationlinesinterval.push_back(rotationlines[i][1]+rotationlines[i][3]-rotationlines[i+1][1]-rotationlines[i+1][3]);
            rotationlinesinterval.push_back(a);
        }
/*
//未旋转过的线计算间隔
        for (int i = 0; i < linesRansac.size() - 1; i++) {
            //Vec2i a(round(ceres::abs(rotationlines[i][1]+rotationlines[i][3]-rotationlines[i+1][1]-rotationlines[i+1][3])),i+1);//利用平行线之间的距离
            int x1 = linesRansac[i][0], y1 = linesRansac[i][1], x2 = linesRansac[i][2], y2 = linesRansac[i][3], x3 = linesRansac[
                    i + 1][0], y3 = linesRansac[i + 1][1], x4 = linesRansac[i + 1][2], y4 = linesRansac[i + 1][3];
//以前的距离是点到直线的距离
//            Vec2i a(round(ceres::abs(
//                    ((y4 - y3) * (x1 + x2) / 2 - (x4 - x3) * (y1 + y2) / 2 + y4 * (x4 - x3) - (y4 - y3) * x4) /
//                    ceres::sqrt(pow((y4 - y3), 2) + pow((x4 - x3), 2)))), i + 1);


            Vec2i a;
            if(x1>x2 && x3>x4) {
                a=Vec2i(ceres::abs(y3-y1), i + 1);
            }
            if(x1>x2 && x4>x3) {
                a=Vec2i(ceres::abs(y4-y1), i + 1);
            }
            if(x2>x1 && x3>x4) {
                a=Vec2i(ceres::abs(y3-y2), i + 1);
            }
            if(x2>x1 && x4>x3) {
                a=Vec2i(ceres::abs(y4-y2), i + 1);
            }

            //Vec2i a(round(ceres::abs((y1+y2)/2)),i+1);
            //rotationlinesinterval.push_back(rotationlines[i][1]+rotationlines[i][3]-rotationlines[i+1][1]-rotationlines[i+1][3]);
            rotationlinesinterval.push_back(a);
        }
*/



//对间隔排序
        sort(rotationlinesinterval.begin(), rotationlinesinterval.end(), [](Vec2i a, Vec2i b) { return a[0] > b[0]; });

/*
        int threadhold;
        double pre = 10000000000;
        for (int i = 1; i < rotationlinesinterval.size() - 2; i++) {
            double meana = 0, meanb = 0, sigmaa = 0, sigmab = 0, absigma = 0;
            for (int j = i - 1; j >= 0; j--) {
                meana = meana + rotationlinesinterval[j][0];
            }
            meana = meana / i;
            for (int j = i + 1; j < rotationlinesinterval.size(); j++) {
                meanb = meanb + rotationlinesinterval[j][0];
            }
            meanb = meanb / (rotationlinesinterval.size() - 1 - i);
            for (int j = i - 1; j >= 0; j--) {
                sigmaa = sigmaa + pow((rotationlinesinterval[j][0] - meana), 2);
            }
            for (int j = i + 1; j < rotationlinesinterval.size(); j++) {
                sigmab = sigmab + pow((rotationlinesinterval[j][0] - meanb), 2);
            }
            absigma = pow((meana - meanb), 2);

            if (sigmaa + sigmab - 2 * absigma < pre) {
                threadhold = i;
            }
            //pre=(sigmaa+sigmab)/absigma;
            pre = sigmaa + sigmab - 2 * absigma;
        }
        cout << "Threadhold = " << threadhold << endl;
*/




//直线聚类
        double data[rotationlinesinterval.size()];

        for (int i = 0; i < rotationlinesinterval.size(); i++) {
            cout << "rotationlinesinterval = " << rotationlinesinterval[i][0] << endl;


            data[i] = rotationlinesinterval[i][0];
        }

        const int size = rotationlines.size(); //Number of samples
        const int dim = 1;   //Dimension of feature
        const int cluster_num = 2; //Cluster number

        KMeans *kmeans = new KMeans(dim, cluster_num);
        int *labels = new int[size];
        kmeans->SetInitMode(KMeans::InitUniform);
        kmeans->Cluster(data, size, labels);

        int clusternum1 = 0;
        int clusternum2 = 0;
        double clustersum1 = 0;
        double clustersum2 = 0;
        for (int i = 0; i < size; ++i) {
            printf("%f, belongs to %d cluster\n", data[i * dim + 0], labels[i]);
            if (labels[i] == 0) {
                clustersum1 = clustersum1 + data[i * dim];
                clusternum1 = clusternum1 + 1;
            }
            if (labels[i] == 1) {
                clustersum2 = clustersum2 + data[i * dim];
                clusternum2 = clusternum2 + 1;
            }

        }

        double averagedis = 0, averagedis1 = clustersum1 / clusternum1, averagedis2 = clustersum2 / clusternum2;
//averagedis有加权,短的直线间的距离权值大一些。
        //if(averagedis1>=averagedis2) {
        //    averagedis = (clusternum1*averagedis1+clusternum2*averagedis2)/(clusternum1+clusternum2);
        //}
        //if(averagedis1<averagedis2) {
        //    averagedis = (clusternum2*averagedis2+clusternum1*averagedis1)/(clusternum1+clusternum2);
        //}


        if (averagedis1 >= averagedis2) {
            averagedis = (averagedis1 + 3 * averagedis2) / 4;
        }
        if (averagedis1 < averagedis2) {
            averagedis = (3 * averagedis2 + averagedis1) / 4;//直线的数量基本固定在200根
        }
        cout << "averagedis = " << averagedis << endl;


        delete[]labels;
        delete kmeans;


        //int intervalnum=4;//画几簇线
        int intervalnum = 4;
        /* for(int i=0;i<rotationlinesinterval.size();i++)
        {
            //if(rotationlinesinterval[1][0]/rotationlinesinterval[i+1][0]>=6)
                if(rotationlinesinterval[i][0]/rotationlinesinterval[i+1][0]>=2*rotationlinesinterval[i+1][0]/rotationlinesinterval[i+2][0])
            {
                intervalnum=i;
                break;
            }
        }
    */
        cout << "intervalnum=" << intervalnum << endl;


        int interval[intervalnum];
        for (int i = 0; i < intervalnum; i++) {
            interval[i] = rotationlinesinterval[i][1];
            //if(i>=intervalnum)
            //{
            //    interval[i]=100;
            //}
        }
        //int interval[5]={rotationlinesinterval[0][1],rotationlinesinterval[1][1],rotationlinesinterval[2][1],rotationlinesinterval[3][1],rotationlinesinterval[4][1]};
        //int interval[5]={rotationlinesinterval[0][2], rotationlinesinterval[1][2], rotationlinesinterval[2][2], rotationlinesinterval[3][2], rotationlinesinterval[4][2]};
        //sort(interval,interval+4,[](int a, int b){return a<b;});
        sort(interval, interval + intervalnum - 1, [](int a, int b) { return a < b; });

        //for(int i=0;i<=3;i++)
        //{
        //    cout<<"interval ="<<interval[i]<<endl;
        //}

        for (int i = 0; i < intervalnum; i++) {
            cout << "interval =" << interval[i] << endl;
        }



        /*
        vector<Vec<double,5> > newrotationlines;

        for(vector<Vec4i>::iterator iterator1=rotationlines.begin(); iterator1!=rotationlines.end();iterator1++)
        {
            int o = iterator1-rotationlines.begin();
            newrotationlines[o][0]=(*iterator1)[0];
            newrotationlines[o][1]=(*iterator1)[1];
            newrotationlines[o][2]=(*iterator1)[2];
            newrotationlines[o][3]=(*iterator1)[3];
            newrotationlines[o][4]=((*iterator1)[3]-(*iterator1)[1])/((*iterator1)[2]-(*iterator1)[0]);
        }




        ceres::Problem problem;

        for (auto item1:newrotationlines) {
            //for (vector<Vec4i>::iterator item1=linesRansac.begin();item1 != linesRansac.end();item1++) {
            for (auto item2:newrotationlines) {
                //for (vector<Vec4i>::iterator item2 = item1; item2 != linesRansac.end();item2++) {
                problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<HomographyParallel, 1, 1, 1>(
                                new HomographyParallel(OnLinePoint(item1[0], item1[1]),
                                                          OnLinePoint(item1[2], item1[3]),
                                                          OnLinePoint(item2[0], item2[1]),
                                                          OnLinePoint(item2[2], item2[3]))), new CauchyLoss(0.5),
                        &item1[4], &item2[4]);
            }
        }



        Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << "theta=" << theta << std::endl;
        */











/*平均值找直线间隔的方法


        double averagedis = 0, predis1 = 1;
        for (std::vector<Vec4i>::iterator iter = rotationlines.begin();
             iter != rotationlines.end(); iter++)//rota 和 rans 一一对应
        {
            double x1 = (*iter)[0], y1 = (*iter)[1], x2 = (*iter)[2], y2 = (*iter)[3];

            //double dis=ceres::abs(((y4-y3)*(x1+x2)/2-(x4-x3)*(y1+y2)/2+y4*(x4-x3)-(y4-y3)*x4)/ceres::sqrt(pow((y4-y3),2)+pow((x4-x3),2)));
            //double dis=(y1+y2);
            double dis = (y2);

//矫正直线拉直
            double mid=(y1-y2)*0.5;
            cout<<"mid= "<<mid<<endl;

            if(y1>y2)
            {
                y1=y1-ceres::abs(mid);
                y2=y2+ceres::abs(mid);
            }
            if(y2>y1)
            {
                y2=y2-ceres::abs(mid);
                y1=y1+ceres::abs(mid);
            }





            if (iter - rotationlines.begin() != 0) {
                averagedis = predis1 - dis + averagedis;//predis-dis 下面的线y值减去上面的线y值
            }
            predis1 = dis;
        }

        //averagedis = 2*averagedis / (rotationlines.size() - 1);//直线阈值
        //cout << "averagedis = " << averagedis << endl;



        averagedis = 2*averagedis / (rotationlines.size() - 1);//直线阈值
        cout << "averagedis = " << averagedis << endl;

*/






        double predis = 1;
        int colorstep = 1;


        int rotationlinenumber[7]={0};
        int flagmemory = 0;//有突变为0不记录 无突变帧间连续要记录

        int lineclusternum=0;
        double ycomp1 = 0, ycomp2=0, ycomp3=0, ycomp4=0, ycomp5=0, ycomp6=0, ycomp7=0;
        int green=0, blue=0, yellow=0, pink=0, cyan=0;
        std::cout<<"frame="<<frame<<std::endl;

             for (std::vector<Vec4i>::iterator iter = rotationlines.begin();
              iter != rotationlines.end() - 1; iter++)//rota 和 rans 一一对应 最后采用了反变换不是对应
           {
             double x1 = (*iter)[0], y1 = (*iter)[1], x2 = (*iter)[2], y2 = (*iter)[3];
             double dis = (y2);
             /////////////////////////////////////////////////////////////////调整阈值线断点间隔
             //if(predis-dis>35 && iter-rotationlines.begin()!=0)
             if (predis - dis > averagedis && iter - rotationlines.begin() != 0) {
                 colorstep=colorstep+1;
                 if (colorstep == 2) {
                     rotationlinenumber[0] = iter - rotationlines.begin();//减1是因为下标是从0开始
                     lineclusternum=2;
                     cout<<"rotationlinenumber[0]="<<rotationlinenumber[0]<<endl;
                 }
                 if (colorstep == 3) {
                     rotationlinenumber[1] = iter - rotationlines.begin();
                     lineclusternum=3;
                     cout<<"rotationlinenumber[1]="<<rotationlinenumber[1]<<endl;
                 }
                 if (colorstep == 4) {
                     rotationlinenumber[2] = iter - rotationlines.begin();
                     lineclusternum=4;
                     cout<<"rotationlinenumber[2]="<<rotationlinenumber[2]<<endl;
                 }
                 if (colorstep == 5) {
                     rotationlinenumber[3] = iter - rotationlines.begin();
                     lineclusternum=5;
                     cout<<"rotationlinenumber[3]="<<rotationlinenumber[3]<<endl;
                 }
                 if (colorstep == 6) {
                     rotationlinenumber[4] = iter - rotationlines.begin();
                     lineclusternum=6;
                     cout<<"rotationlinenumber[4]="<<rotationlinenumber[4]<<endl;
                 }
                 if (colorstep == 7) {
                     rotationlinenumber[5] = iter - rotationlines.begin();
                     lineclusternum=7;
                     cout<<"rotationlinenumber[5]="<<rotationlinenumber[5]<<endl;
                 }
             }
             predis = dis;
           }


            rotationlinenumber[lineclusternum - 1] = rotationlines.size();
            colorstep = 1;




            //double ycomp1 = 0, ycomp2=0, ycomp3=0, ycomp4=0, ycomp5=0, ycomp6=0, ycomp7=0;

            if(rotationlinenumber[0] != 0) {
                for (int i = 0; i < rotationlinenumber[0]; i++) {
                    ycomp1 = ycomp1 + rotationlines[i][3];
                }
                ycomp1 = ycomp1 / rotationlinenumber[0];
            }
            if(rotationlinenumber[1] != 0) {
                for (int i = rotationlinenumber[0]; i < rotationlinenumber[1]; i++) {
                    ycomp2 = ycomp2 + rotationlines[i][3];
                }
                ycomp2 = ycomp2 / (rotationlinenumber[1] - rotationlinenumber[0]);
            }
            if(rotationlinenumber[2] != 0) {
                for (int i = rotationlinenumber[1]; i < rotationlinenumber[2]; i++) {
                    ycomp3 = ycomp3 + rotationlines[i][3];
                }
                ycomp3 = ycomp3 / (rotationlinenumber[2] - rotationlinenumber[1]);
            }
            if(rotationlinenumber[3] != 0) {
                for (int i = rotationlinenumber[2]; i < rotationlinenumber[3]; i++) {
                    ycomp4 = ycomp4 + rotationlines[i][3];
                }
                ycomp4 = ycomp4 / (rotationlinenumber[3] - rotationlinenumber[2]);
            }
            if(rotationlinenumber[4] != 0) {
                for (int i = rotationlinenumber[3]; i < rotationlinenumber[4]; i++) {
                    ycomp5 = ycomp5 + rotationlines[i][3];
                }
                ycomp5 = ycomp5 / (rotationlinenumber[4] - rotationlinenumber[3]);
            }
            if(rotationlinenumber[5] != 0) {
                for (int i = rotationlinenumber[4]; i < rotationlinenumber[5]; i++) {
                    ycomp6 = ycomp6 + rotationlines[i][3];
                }
                ycomp6 = ycomp6 / (rotationlinenumber[5] - rotationlinenumber[4]);
            }
            if(rotationlinenumber[6] != 0) {
                for (int i = rotationlinenumber[5]; i < rotationlinenumber[6]; i++) {
                    ycomp7 = ycomp7 + rotationlines[i][3];
                }
                ycomp7 = ycomp7 / (rotationlinenumber[6] - rotationlinenumber[5]);
            }





        if(frame!=1){

            vector<double> comp1, comp2, comp3, comp4, comp5, comp6, comp7;
            std::vector<double>::iterator smallest1, smallest2, smallest3, smallest4, smallest5, smallest6, smallest7;
            int order1, order2, order3, order4, order5, order6, order7;
            if(rotationlinenumber[0] != 0) {
                for (int i = 0; i < linemark.size(); i++) {
                    double temp = ceres::abs(linemark[i] - ycomp1);
                    comp1.push_back(temp);
                }
                smallest1 = std::min_element(std::begin(comp1), std::end(comp1));
                order1 = smallest1 - std::begin(comp1);
                std::cout << "smallestorder1=" << order1 << endl;
            }
            if(rotationlinenumber[1] != 0) {
                for (int i = 0; i < linemark.size(); i++) {
                    double temp = ceres::abs(linemark[i] - ycomp2);
                    comp2.push_back(temp);
                }
                smallest2 = std::min_element(std::begin(comp2), std::end(comp2));
                order2 = smallest2 - std::begin(comp2);
                std::cout << "smallestorder2=" << order2 << endl;
            }
            if(rotationlinenumber[2] != 0) {
                for (int i = 0; i < linemark.size(); i++) {
                    double temp = ceres::abs(linemark[i] - ycomp3);
                    comp3.push_back(temp);
                }
                smallest3 = std::min_element(std::begin(comp3), std::end(comp3));
                order3 = smallest3 - std::begin(comp3);
                std::cout << "smallestorder3=" << order3 << endl;
            }
            if(rotationlinenumber[3] != 0) {
                for (int i = 0; i < linemark.size(); i++) {
                    double temp = ceres::abs(linemark[i] - ycomp4);
                    comp4.push_back(temp);
                }
                smallest4 = std::min_element(std::begin(comp4), std::end(comp4));
                order4 = smallest4 - std::begin(comp4);
                std::cout << "smallestorder4=" << order4 << endl;
            }
            if(rotationlinenumber[4] != 0) {
                for (int i = 0; i < linemark.size(); i++) {
                    double temp = ceres::abs(linemark[i] - ycomp5);
                    comp5.push_back(temp);
                }
                smallest5 = std::min_element(std::begin(comp5), std::end(comp5));
                order5 = smallest5 - std::begin(comp5);
                std::cout << "smallestorder5=" << order5 << endl;
            }
            if(rotationlinenumber[5] != 0) {
                for (int i = 0; i < linemark.size(); i++) {
                    double temp = ceres::abs(linemark[i] - ycomp6);
                    comp6.push_back(temp);
                }
                smallest6 = std::min_element(std::begin(comp6), std::end(comp6));
                order6 = smallest6 - std::begin(comp6);
                std::cout << "smallestorder6=" << order6 << endl;
            }
            if(rotationlinenumber[6] != 0) {
                for (int i = 0; i < linemark.size(); i++) {
                    double temp = ceres::abs(linemark[i] - ycomp7);
                    comp7.push_back(temp);
                }
                smallest7 = std::min_element(std::begin(comp7), std::end(comp7));
                order7 = smallest7 - std::begin(comp7);
                std::cout << "smallestorder7=" << order7 << endl;
            }

            //std::vector<double>::iterator smallest1 = std::min_element(std::begin(comp1), std::end(comp1));
            //int order1 = smallest1 - std::begin(comp1);

            //sort(comp.begin(),comp.end(),[](double a, double b){return a<b;});



            //std::cout << "smallestorder1=" << order1 << endl;
            if (order1 == 0) {//红色线向下走 与 原来存储的线条 仍然还是与 0号下标的线最接近 红线下去存储信息马上就更新为当前最新的了 所以不变颜色按聚类来
                colorstep = 1;
            }
            if (order1 == 1) {//红线丢掉 选颜色的时候直接进绿色
                colorstep = 2;
            }
            if (order1 == 2) {
                colorstep = 3;
            }
            if (order1 == 3) {
                colorstep = 4;
            }
            if (order1 == 4) {
                colorstep = 5;
            }

            if(rotationlinenumber[1] != 0) {

                if(order2 == 1)
                {
                    green=0;
                }
                if(order2 != 1)//绿线丢掉 红线在 选颜色进去的时候绿线加一变蓝线
                {
                    green=1;
                }
                //if(order2 == 2)
                //{
                //    green=1;
                //}
                if(order2 == 3)
                {
                    green=2;
                }
            }

            if(rotationlinenumber[2] != 0) {

                if(order3 == 2)
                {
                    blue=0;
                }
                //if(order3 != 2)
                //{
                //    blue=1;
                //}
                if(order3 == 3)
                {
                    blue=1;
                }
                if(order3 == 4)
                {
                    blue=2;
                }
            }




        }


        if(colorstep==1 && green==0) {//红线未丢或者红线往下走          红线存在的情况下绿线未丢
            flagmemory = 1;
        }

        //做一下选择
        CvScalar color;


        std::cout<<"colorstep="<<colorstep<<std::endl;
        if (colorstep == 1) {
            color = CV_RGB(255, 0, 0);
        }
        if (colorstep == 2) {
            color = CV_RGB(0, 255, 0);
        }
        if (colorstep == 3) {
            color = CV_RGB(0, 0, 255);
        }
        if (colorstep == 4) {
            color = CV_RGB(255, 255, 0);
        }
        if (colorstep == 5) {
            color = CV_RGB(255, 0, 255);
        }
        if (colorstep == 6) {
            color = CV_RGB(0, 255, 255);
        }
        if (colorstep == 7) {
            color = CV_RGB(255, 255, 255);
        }



        //linemark.clear();















        int rotationflag=1;
        //int lineclusternum=0;
        for (std::vector<Vec4i>::iterator iter = rotationlines.begin();
             iter != rotationlines.end() - 1; iter++)//rota 和 rans 一一对应 最后采用了反变换不是对应
        {
            //double x1 = (*iter)[0], y1 = (*iter)[1], x2 = (*iter)[2], y2 = (*iter)[3], x3 = (*(iter + 1))[0], y3 = (*(
            //        iter + 1))[1], x4 = (*(iter + 1))[2], y4 = (*(iter + 1))[3];
            double x1 = (*iter)[0], y1 = (*iter)[1], x2 = (*iter)[2], y2 = (*iter)[3];

            double x1origin = (*iter)[0], y1origin = (*iter)[1], x2origin = (*iter)[2], y2origin = (*iter)[3];


            double mid=(y2);
//把直线矫正平行 利用前后两点平均间隔做计算
/*            double mid=(y1-y2)*0.5;
            cout<<"mid= "<<mid<<endl;




            if(y1>y2)
            {
                y1=y1-ceres::abs(mid);
                y2=y2+ceres::abs(mid);
            }
            if(y2>y1)
            {
                y2=y2-ceres::abs(mid);
                y1=y1+ceres::abs(mid);
            }
            //y1=y1+mid;
            //y2=y2-mid;
*/



            //double dis=ceres::abs(((y4-y3)*(x1+x2)/2-(x4-x3)*(y1+y2)/2+y4*(x4-x3)-(y4-y3)*x4)/ceres::sqrt(pow((y4-y3),2)+pow((x4-x3),2)));
            //double dis=(y1+y2);
            double dis = (y2);


            cout << "distance = " << dis << endl;
            cout << "predistance = " << predis << endl;
            /////////////////////////////////////////////////////////////////调整阈值线断点间隔
            //if(predis-dis>35 && iter-rotationlines.begin()!=0)
            if (predis - dis > averagedis && iter - rotationlines.begin() != 0) {
                if (rotationflag == 1) {
                //    color = CV_RGB(255, 0, 0);
                    //if() {
                        //rotationlinenumber[0] = iter - rotationlines.begin()-1;//默认是红色 然后进来要变色 减去1表示红色
                    //cout<<"rotationlinenumber[0]="<<rotationlinenumber[0]<<endl;
                    //}
                    rotationflag=0;
                }
                colorstep=colorstep+1;


                if (colorstep == 2) {
                    color = CV_RGB(0, 255, 0);
                    //rotationlinenumber[0] = iter - rotationlines.begin();//减1是因为下标是从0开始
                    //cout<<"rotationlinenumber[0]="<<rotationlinenumber[0]<<endl;
                    //lineclusternum=2;
                    if(green==1)//红线绿线不能同时丢掉的情况
                    {
                        colorstep++;
                    }
                }
                if (colorstep == 3) {
                    color = CV_RGB(0, 0, 255);
                    //rotationlinenumber[1] = iter - rotationlines.begin();
                    //cout<<"rotationlinenumber[1]="<<rotationlinenumber[1]<<endl;
                    //lineclusternum=3;
                    //if(blue==1)
                    //{
                    //    colorstep++;
                    //}
                }
                if (colorstep == 4) {
                    color = CV_RGB(255, 255, 0);
                    //rotationlinenumber[2] = iter - rotationlines.begin();
                    //cout<<"rotationlinenumber[2]="<<rotationlinenumber[2]<<endl;
                    //lineclusternum=4;
                }
                if (colorstep == 5) {
                    color = CV_RGB(255, 0, 255);
                    //rotationlinenumber[3] = iter - rotationlines.begin();
                    //cout<<"rotationlinenumber[3]="<<rotationlinenumber[3]<<endl;
                    //lineclusternum=5;
                }
                if (colorstep == 6) {
                    color = CV_RGB(0, 255, 255);
                    //rotationlinenumber[4] = iter - rotationlines.begin();
                    //cout<<"rotationlinenumber[4]="<<rotationlinenumber[4]<<endl;
                    //lineclusternum=5;
                }
                if (colorstep == 7) {
                    color = CV_RGB(255, 255, 255);
                    //rotationlinenumber[5] = iter - rotationlines.begin();
                    //cout<<"rotationlinenumber[5]="<<rotationlinenumber[5]<<endl;
                    //lineclusternum=5;
                }
                //colorstep++;
            }
            predis = dis;




            //Eigen::Vector3d begin((*iter)[0], (*iter)[1], 1), end((*iter)[2], (*iter)[3], 1), newbegin, newend;
//原始线 单应矫正后没有拉平行
            Eigen::Vector3d begin(x1origin, y1origin, 1), end(x2origin, y2origin, 1), newbegin, newend;
//原始线 单应矫正后拉平行
            //Eigen::Vector3d begin(x1, y1, 1), end(x2, y2, 1), newbegin, newend;
            newbegin = H.inverse() * begin;
            newend = H.inverse() * end;
            //std::cout<<"newbegin="<<newbegin<<std::endl;
            //std::cout<<"newend="<<newend<<std::endl;
            newbegin = newbegin / newbegin[2];
            newend = newend / newend[2];
            Vec4i newline(round(newbegin[0]), round(newbegin[1]), round(newend[0]), round(newend[1]));

            //cout<<"间隔= "<<iter-rotationlines.begin()<<endl;





            line(srctemp, cv::Point(newline[0], newline[1]), cv::Point(newline[2], newline[3]), color, 0.5);//先在图上划了线在对应过去




/*
        Eigen::Vector3d begin((*iter)[0],(*iter)[1],1),end((*iter)[2],(*iter)[3],1),newbegin,newend;
        newbegin=H.inverse()*begin;
        newend=H.inverse()*end;
        //std::cout<<"newbegin="<<newbegin<<std::endl;
        //std::cout<<"newend="<<newend<<std::endl;
        newbegin=newbegin/newbegin[2];
        newend=newend/newend[2];
        Vec4i newline(round(newbegin[0]),round(newbegin[1]),round(newend[0]),round(newend[1]));

        cout<<"间隔= "<<iter-rotationlines.begin()<<endl;
        //if(iter-rotationlines.begin()==interval[0]-1)


        if(iter-rotationlines.begin()<=interval[0]-1)//换成==只画一根线 从下往上红绿蓝黄粉
        {
            line(srctemp, cv::Point(newline[0], newline[1]), cv::Point(newline[2], newline[3]), CV_RGB(255, 0, 0), 0.5);
        }
        //else if(iter-rotationlines.begin()<=interval[1]-1)
        else if(interval[0]-1 <=iter-rotationlines.begin() && iter-rotationlines.begin()<=interval[1]-1)
        //if(interval[0]-1<iter-rotationlines.begin() && iter-rotationlines.begin()<=interval[1]-1)
       {
            line(srctemp, cv::Point(newline[0], newline[1]), cv::Point(newline[2], newline[3]), CV_RGB(0, 255, 0), 0.5);
        }
        //else if(iter-rotationlines.begin()<=interval[2]-1)
        else if(interval[1]-1 <=iter-rotationlines.begin() && iter-rotationlines.begin()<=interval[2]-1)
        //if( interval[1]-1<iter-rotationlines.begin() && iter-rotationlines.begin()<=interval[2]-1)
        {
            line(srctemp, cv::Point(newline[0], newline[1]), cv::Point(newline[2], newline[3]), CV_RGB(0, 0, 255), 0.5);
        }
        else if(interval[2]-1<=iter-rotationlines.begin() && iter-rotationlines.begin()<=interval[3]-1)
        //if(interval[2]-1<iter-rotationlines.begin() && iter-rotationlines.begin()<=interval[3]-1)
        {
            line(srctemp, cv::Point(newline[0], newline[1]), cv::Point(newline[2], newline[3]), CV_RGB(255, 255, 0), 0.5);
        }
       else
        //else if(iter-rotationlines.begin()<=interval[4]-1)
        //if( interval[3]-1 < iter-rotationlines.begin()&& iter-rotationlines.begin()<=interval[4]-1) {
        {    line(srctemp, cv::Point(newline[0], newline[1]), cv::Point(newline[2], newline[3]), CV_RGB(255, 0, 255),
                 0.5);
        }
         //else if(iter-rotationlines.begin()<=interval[5]-1)
        //if( interval[4]-1 < iter-rotationlines.begin()&& iter-rotationlines.begin()<=interval[5]-1) {
       // else
        //{    line(srctemp, cv::Point(newline[0], newline[1]), cv::Point(newline[2], newline[3]), CV_RGB(0, 255, 255),
        //         0.5);
        //}
        /*else if(iter-rotationlines.begin()<=interval[6]-1)
        //if( interval[5]-1 < iter-rotationlines.begin()&& iter-rotationlines.begin()<=interval[6]-1) {
        {    line(srctemp, cv::Point(newline[0], newline[1]), cv::Point(newline[2], newline[3]), CV_RGB(0, 0, 0),
                 0.5);
        }
        else if(iter-rotationlines.begin()<=interval[7]-1)
        //if( interval[6]-1 < iter-rotationlines.begin()&& iter-rotationlines.begin()<=interval[7]-1) {

        {    line(srctemp, cv::Point(newline[0], newline[1]), cv::Point(newline[2], newline[3]), CV_RGB(120, 120, 120),
                 0.5);
        }
        */
        }
 /*       double line1=0, line2=0, line3=0, line4=0, line5=0, line6=0, line7 = 0;
        //if(flagmemory==1) {
        //if(frame!=1) {

            rotationlinenumber[lineclusternum - 1] = rotationlines.size();
            for (int i = 0; i <= rotationlines.size(); i++) {

                if (i < rotationlinenumber[0] && rotationlinenumber[0] != 0) {
                    line1 = line1 + rotationlines[i][3];
                }


                if (i >= rotationlinenumber[0] && i < rotationlinenumber[1] && rotationlinenumber[1] != 0) {
                    line2 = line2 + rotationlines[i][3];
                }


                if (i >= rotationlinenumber[1] && i < rotationlinenumber[2] && rotationlinenumber[2] != 0) {
                    line3 = line3 + rotationlines[i][3];
                }


                if (i >= rotationlinenumber[2] && i < rotationlinenumber[3] && rotationlinenumber[3] != 0) {
                    line4 = line4 + rotationlines[i][3];
                }


                if (i >= rotationlinenumber[3] && i < rotationlinenumber[4] && rotationlinenumber[4] != 0) {
                    line5 = line5 + rotationlines[i][3];
                }


                if (i >= rotationlinenumber[4] && i < rotationlinenumber[5] && rotationlinenumber[5] != 0) {
                    line6 = line6 + rotationlines[i][3];
                }


                if (i >= rotationlinenumber[5] && i < rotationlinenumber[6] && rotationlinenumber[6] != 0) {
                    line7 = line7 + rotationlines[i][3];
                }

            }
            std::cout<<"!3.6"<<endl;
            if (rotationlinenumber[0] != 0) {
                line1 = line1 / rotationlinenumber[0];
                //line1 = (2 * linemark[0] + line1) / 3;
                cout << "line1=" << line1 << endl;
                //linemark.push_back(line1);
            }
            if (rotationlinenumber[1] != 0) {
                line2 = line2 / (rotationlinenumber[1] - rotationlinenumber[0]);
                //line2 = (2 * linemark[1] + line2) / 3;
                cout << "line2=" << line2 << endl;
                //linemark.push_back(line2);
            }
            if (rotationlinenumber[2] != 0) {
                line3 = line3 / (rotationlinenumber[2] - rotationlinenumber[1]);
                //line3 = (2 * linemark[2] + line3) / 3;
                //linemark.push_back(line3);
                cout << "line3=" << line3 << endl;
            }
            if (rotationlinenumber[3] != 0) {
                line4 = line4 / (rotationlinenumber[3] - rotationlinenumber[2]);
                //line4 = (2 * linemark[3] + line4) / 3;
                //linemark.push_back(line4);
                cout << "line4=" << line4 << endl;
            }
            if (rotationlinenumber[4] != 0) {
                line5 = line5 / (rotationlinenumber[4] - rotationlinenumber[3]);
                //line5 = (2 * linemark[4] + line5) / 3;
                //linemark.push_back(line5);
                cout << "line5=" << line5 << endl;
            }
            if (rotationlinenumber[5] != 0) {
                line6 = line6 / (rotationlinenumber[5] - rotationlinenumber[4]);
                //line6 = (2 * linemark[5] + line6) / 3;
                //linemark.push_back(line6);
                cout << "line6=" << line6 << endl;
            }
            if (rotationlinenumber[6] != 0) {
                line7 = line7 / (rotationlinenumber[6] - rotationlinenumber[5]);
                //line7 = (2 * linemark[6] + line7) / 3;
                //linemark.push_back(line7);
                cout << "line7=" << line7 << endl;
            }


*/
        //}

        if(frame==1) {

            /*linemark.push_back(line1);
            //std::cout<<"!3.7"<<endl;
            linemark.push_back(line2);
            linemark.push_back(line3);
            linemark.push_back(line4);
            linemark.push_back(line5);
            linemark.push_back(line6);
            linemark.push_back(line7);*/
            linemark.push_back(ycomp1);
            //std::cout<<"!3.7"<<endl;
            linemark.push_back(ycomp2);
            linemark.push_back(ycomp3);
            linemark.push_back(ycomp4);
            linemark.push_back(ycomp5);
            linemark.push_back(ycomp6);
            linemark.push_back(ycomp7);

        }

        else{
            /*if(flagmemory==1) {//允许更新
                line1 = (2 * linemark[0] + line1) / 3;
                linemark[0] = line1;
                line2 = (2 * linemark[1] + line2) / 3;
                linemark[1] = line2;
                line3 = (2 * linemark[2] + line3) / 3;
                linemark[2] = line3;
                line4 = (2 * linemark[3] + line4) / 3;
                linemark[3] = line4;
                line5 = (2 * linemark[4] + line5) / 3;
                linemark[4] = line5;
                line6 = (2 * linemark[5] + line6) / 3;
                linemark[5] = line6;
                line7 = (2 * linemark[6] + line7) / 3;
                linemark[6] = line7;
            }
            if(flagmemory==0) {//小权重更新
                line1 = (19 * linemark[0] + line1) / 20;
                linemark[0] = line1;
                line2 = (19 * linemark[1] + line2) / 20;
                linemark[1] = line2;
                line3 = (19 * linemark[2] + line3) / 20;
                linemark[2] = line3;
                line4 = (19 * linemark[3] + line4) / 20;
                linemark[3] = line4;
                line5 = (19 * linemark[4] + line5) / 20;
                linemark[4] = line5;
                line6 = (19 * linemark[5] + line6) / 20;
                linemark[5] = line6;
                line7 = (19 * linemark[6] + line7) / 20;
                linemark[6] = line7;
            }*/
            if(flagmemory==1) {//允许更新
                ycomp1 = (2 * linemark[0] + ycomp1) / 3;
                linemark[0] = ycomp1;
                ycomp2 = (2 * linemark[1] + ycomp2) / 3;
                linemark[1] = ycomp2;
                ycomp3 = (2 * linemark[2] + ycomp3) / 3;
                linemark[2] = ycomp3;
                ycomp4 = (2 * linemark[3] + ycomp4) / 3;
                linemark[3] = ycomp4;
                ycomp5 = (2 * linemark[4] + ycomp5) / 3;
                linemark[4] = ycomp5;
                ycomp6 = (2 * linemark[5] + ycomp6) / 3;
                linemark[5] = ycomp6;
                ycomp7 = (2 * linemark[6] + ycomp7) / 3;
                linemark[6] = ycomp7;
            }
            /* if(flagmemory==0) {//小权重更新
                 line1 = (19 * linemark[0] + line1) / 20;
                 linemark[0] = line1;
                 line2 = (19 * linemark[1] + line2) / 20;
                 linemark[1] = line2;
                 line3 = (19 * linemark[2] + line3) / 20;
                 linemark[2] = line3;
                 line4 = (19 * linemark[3] + line4) / 20;
                 linemark[3] = line4;
                 line5 = (19 * linemark[4] + line5) / 20;
                 linemark[4] = line5;
                 line6 = (19 * linemark[5] + line6) / 20;
                 linemark[5] = line6;
                 line7 = (19 * linemark[6] + line7) / 20;
                 linemark[6] = line7;
             }*/
        }















/*        vector<Vec4i> newparallel;
        for(int i=0;i<7;i++)
        {
            if (rotationlinenumber[i]!=0)
            {
                //Eigen::Vector3d begin((*iter)[0], (*iter)[1], 1), end((*iter)[2], (*iter)[3], 1), newbegin, newend;
                //newbegin = H.inverse() * begin;
                //newend = H.inverse() * end;
                ////std::cout<<"newbegin="<<newbegin<<std::endl;
                ////std::cout<<"newend="<<newend<<std::endl;
                //newbegin = newbegin / newbegin[2];
                //newend = newend / newend[2];
                //Vec4i newline(round(newbegin[0]), round(newbegin[1]), round(newend[0]), round(newend[1]));
                newparallel.push_back(linesRansac[rotationlinenumber[i]]);

                Eigen::Vector3d begin(rotationlines[rotationlinenumber[i]][0], rotationlines[rotationlinenumber[i]][1], 1), end(rotationlines[rotationlinenumber[i]][2], rotationlines[rotationlinenumber[i]][3], 1), newbegin, newend;
                newbegin = H.inverse() * begin;
                newend = H.inverse() * end;
                //std::cout<<"newbegin="<<newbegin<<std::endl;
                //std::cout<<"newend="<<newend<<std::endl;
                newbegin = newbegin / newbegin[2];
                newend = newend / newend[2];
                Vec4i newline(round(newbegin[0]), round(newbegin[1]), round(newend[0]), round(newend[1]));

                line(srctemp, cv::Point(newline[0], newline[1]), cv::Point(newline[2], newline[3]), color, 0.5);//先在图上划了线在对应过去
            }
            cout<<"rotationlinenumber[i]="<<rotationlinenumber[i]<<std::endl;






        }

        ceres::Problem problem;
        for (auto item1:newparallel) {
        //for (vector<Vec4i>::iterator item1=linesRansac.begin();item1 != linesRansac.end();item1++) {
            for (auto item2:newparallel) {
        //for (vector<Vec4i>::iterator item2 = item1; item2 != linesRansac.end();item2++) {
                problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<HomographyCostFunctor, 1, 1>(
                                new HomographyCostFunctor(OnLinePoint(item1[0], item1[1]),
                                                          OnLinePoint(item1[2], item1[3]),
                                                          OnLinePoint(item2[0], item2[1]),
                                                          OnLinePoint(item2[2], item2[3]))), NULL,//new CauchyLoss(0.5),
                        &theta);
            }
        }
        Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << "theta=" << theta << std::endl;

        Ry << ceres::cos(theta/180*pi),0,ceres::sin(theta/180*pi),0,1,0,ceres::sin(-theta/180*pi),0,ceres::cos(theta/180*pi);
        H=K*Ry*K.inverse();
*/




        /* int rotationlinesnumber[5];
         for(std::vector<Vec3i>::iterator iter=rotationlinesinterval.begin() ; iter!=rotationlinesinterval.end() ; iter++)//rota 和 rans 一一对应
         {

             if((*iter)[2]==interval[0])
             {

                 rotationlinesnumber[0]=(*iter)[1];

             }
             if((*iter)[2]==interval[1])
             {
                 rotationlinesnumber[1]=(*iter)[1];
             }
             if((*iter)[2]==interval[2])
             {
                 rotationlinesnumber[1]=(*iter)[1];
             }
             if((*iter)[2]==interval[3])
             {
                 rotationlinesnumber[1]=(*iter)[1];
             }
             if((*iter)[2]==interval[4])
             {
                 rotationlinesnumber[1]=(*iter)[1];
             }


        }

        sort(rotationlinesnumber,rotationlinesnumber+5,[](int a, int b){return a<b ;});

        for (std::vector<Vec<int,5> >::iterator iter=rotationlines.begin() ; iter!=rotationlines.end() ; iter++) {
            if(iter-rotationlines.begin() < rotationlinesnumber[0]) {
                for() {
                    line(srctemp, cv::Point(linesRansac[(*iter)[4]][0], linesRansac[(*iter)[4]][1]),
                         cv::Point(linesRansac[(*iter)[4]][0], linesRansac[(*iter)[4]][1]), CV_RGB(255, 0, 0), 0.5);
                }
            }
            else if(iter-rotationlines.begin() < rotationlinesnumber[1]) {
                line(srctemp, cv::Point(linesRansac[(*iter)[4]][0], linesRansac[(*iter)[4]][1]), cv::Point(linesRansac[(*iter)[4]][0], linesRansac[(*iter)[4]][1]), CV_RGB(0, 255, 0), 0.5);
            }
            else if(iter-rotationlines.begin() < rotationlinesnumber[2]) {
                line(srctemp, cv::Point(linesRansac[(*iter)[4]][0], linesRansac[(*iter)[4]][1]), cv::Point(linesRansac[(*iter)[4]][0], linesRansac[(*iter)[4]][1]), CV_RGB(0, 0, 255), 0.5);
            }
            else if(iter-rotationlines.begin() < rotationlinesnumber[3]) {
                line(srctemp, cv::Point(linesRansac[(*iter)[4]][0], linesRansac[(*iter)[4]][1]), cv::Point(linesRansac[(*iter)[4]][0], linesRansac[(*iter)[4]][1]), CV_RGB(255, 255, 0), 0.5);
            }
            else if(iter-rotationlines.begin() < rotationlinesnumber[4]) {
                line(srctemp, cv::Point(linesRansac[(*iter)[4]][0], linesRansac[(*iter)[4]][1]), cv::Point(linesRansac[(*iter)[4]][0], linesRansac[(*iter)[4]][1]), CV_RGB(0, 255, 255), 0.5);
            }

            //  line(outputImg, etlm, Point2f(0.5 * (item[0] + item[2]), 0.5 * (item[1] + item[3])), CV_RGB(0, 255, 0),
            //      0.5);
        }


    */



        /* double data[clustery.size()];
        for(int i=0;i<clustery.size();i++) {
            data[i] = clustery[i];
        }
       const int size = clustery.size();
        const int dim=1;
        const int cluster_num = 5;

        KMeans* kmeans = new KMeans(dim,cluster_num);
        int* labels = new int[size];
        kmeans->SetInitMode(KMeans::InitUniform);
        kmeans->Cluster(data,size,labels);

        //for(int i = 0; i < size; ++i)
        //{
        //    printf("%f, belongs to %d cluster\n", data[i*dim+0], labels[i]);
        //}

        int tempi=0;
        for (auto lines:linesRansac) {
            if(labels[tempi]==0) {
                line(srctemp, cv::Point(lines[0], lines[1]), cv::Point(lines[2], lines[3]), CV_RGB(0, 255, 0), 0.5);
            }
            if(labels[tempi]==1) {
                line(srctemp, cv::Point(lines[0], lines[1]), cv::Point(lines[2], lines[3]), CV_RGB(255, 255, 0), 0.5);
            }
            if(labels[tempi]==2) {
                line(srctemp, cv::Point(lines[0], lines[1]), cv::Point(lines[2], lines[3]), CV_RGB(0, 255, 255), 0.5);
            }
            if(labels[tempi]==3) {
                line(srctemp, cv::Point(lines[0], lines[1]), cv::Point(lines[2], lines[3]), CV_RGB(255, 0, 0), 0.5);
            }
            if(labels[tempi]==4) {
                line(srctemp, cv::Point(lines[0], lines[1]), cv::Point(lines[2], lines[3]), CV_RGB(0, 0, 255), 0.5);
            }

            //  line(outputImg, etlm, Point2f(0.5 * (item[0] + item[2]), 0.5 * (item[1] + item[3])), CV_RGB(0, 255, 0),
            //      0.5);
        tempi++;
        }
        //waitKey(3000);

        delete []labels;
        delete kmeans;
    */


        int rows = srctemp.rows;
        int cols = srctemp.cols;
        Eigen::Vector3d leftup(0, 0, 1), leftbottom(0, rows, 1), rightup(cols, 0, 1), rightbottom(cols, rows,
                                                                                                  1), newleftup, newleftbottom, newrightup, newrightbottom;
        //Eigen::Vector3d leftup(0,0,1), leftbottom(0,cols-1,1), rightup(rows-1,0,1), rightbottom(rows-1,cols-1,1),newleftup,newleftbottom,newrightup,newrightbottom;
        newleftup = H * leftup;
        newleftbottom = H * leftbottom;
        newrightbottom = H * rightbottom;
        newrightup = H * rightup;

        newleftup = newleftup / newleftup[2];
        newleftbottom = newleftbottom / newleftbottom[2];
        newrightup = newrightup / newrightup[2];
        newrightbottom = newrightbottom / newrightbottom[2];

        std::initializer_list<double> a = {newleftup[0], newleftbottom[0], newrightup[0], newrightbottom[0]},
                b = {newleftup[1], newleftbottom[1], newrightup[1], newrightbottom[1]};

        for (initializer_list<double>::iterator iter = a.begin(); iter != a.end(); iter++) {
            std::cout << "a = " << *iter << std::endl;
        }
        for (initializer_list<double>::iterator iter = b.begin(); iter != b.end(); iter++) {
            std::cout << "b = " << *iter << std::endl;
        }

        double start1 = min(a), end1 = max(a), start2 = min(b), end2 = max(b);

        std::cout << "start1 min(a)" << start1 << std::endl;
        std::cout << "end1 max(a)" << end1 << std::endl;
        std::cout << "start2 min(b)" << start2 << std::endl;
        std::cout << "end2 max(b)" << end2 << std::endl;


        double k1 = (newleftup(1) - newrightup(1)) / (newleftup(0) - newrightup(0)),
        //        b1=newleftup(1)-k1*newleftup(0),k2=(newleftbottom(1)-newrightbottom(1))/(newleftbottom(0)-newrightbottom(0)),
        //        b2=newleftbottom(1)-k2*newleftbottom(0);
                b1 = newrightup(1) - k1 * newrightup(0), k2 =
                (newleftbottom(1) - newrightbottom(1)) / (newleftbottom(0) - newrightbottom(0)),
                b2 = newrightbottom(1) - k2 * newrightbottom(0);

        //1200


        cout<<"!!!!!!!!!!!!!!!!!!"<<"newrightup(1)-newrightbottom(1)="<<newrightbottom(1)-newrightup(1)<<endl;
        cout<<"!!!!!!!!!!!!!!!!!!"<<"newleftup(1)-newleftbottom(1)="<<newleftup(1)-newleftbottom(1)<<endl;

        double start3 = newrightup(0) - 5000, ystart3 = k1 * start3 + b1, yend3 = k2 * start3 + b2;
        cout << "!4" << endl;
        cout << "round(start3)= " << round(start3) << endl;
        cout << "round(end1)= " << round(end1) << endl;
        cout << "round(yend3)= " << round(yend3) << endl;
        cout << "round(ystart3)= " << round(ystart3) << endl;
        cout << "round(newrightup[0])=" << round(newrightup[0]) << endl;

        //cout<<"round(start3)= "<<round(start3)<<endl;
        //cv::Mat img;

        cv::Mat img(ceres::abs(round(yend3) - round(ystart3)), ceres::abs(round(newrightup(0)) - round(start3)),
                    CV_8UC3, Scalar(0, 0, 0));

        //cv::Mat img(round(end2)-round(start2),round(end1)-round(start1),CV_8UC3,Scalar(255,0,0));

        cout << "round(newrightup[0])-round(start3) =" << round(newrightup[0]) - round(start3) << endl;
        cout << "round(end1)-round(start3) =" << round(end1) - round(start3) << endl;
        cout << "round(yend3)-round(ystart3) =" << round(yend3) - round(ystart3) << endl;

        //resize(img,img,cv::Size(1280,round(2*1280/ceres::abs(round(end1)-round(start3))*ceres::abs(round(yend3)-round(ystart3)))));
        //namedWindow("homography1",WINDOW_FREERATIO);
        //imshow("homography1",img);
        //waitKey(0);

        //imshow("srctemp",srctemp);
        cout << "rows=" << rows << endl;
        cout << "cols=" << cols << endl;


        for (int i = round(start3); i <= round(newrightup(0)); i++)
            //for(int i=round(start1);i<=round(end1);i++)
        {
            for (int j = round(ystart3); j <= round(yend3); j++)
                //for(int j=round(start2);j<=round(end2);j++)
            {
                //cout<<"!!!!!!!!!!!!!!!!!!!!"<<endl;
                Eigen::Vector3d temp(i, j, 1), A;
                A = H.inverse() * temp;
                A(0) = A(0) / A(2);
                A(1) = A(1) / A(2);
                int i2 = round(A(0)), j2 = round(A(1));

                //cout<<"i2="<<i2<<endl;
                //cout<<"j2="<<j2<<endl;

                //cout<<"j-round(ystart3)+1="<<j-round(ystart3)+1<<endl;
                //cout<<"i-round(start3)+1="<<i-round(start3)+1<<endl;
                if (j2 < 0 || i2 < 0 || j2 >= rows || i2 >= cols) {
                    //  img.at<Vec3b>(i-round(start3),j-round(ystart3))[0]=(uchar)0;
                    //  img.at<Vec3b>(i-round(start3),j-round(ystart3))[1]=(uchar)0;
                    //  img.at<Vec3b>(i-round(start3),j-round(ystart3))[2]=(uchar)0;
                    continue;
                }
                img.at<Vec3b>(j - round(ystart3) + 1, i - round(start3) + 1)[0] = srctemp.at<Vec3b>(j2, i2)[0];//行列标签 列为x 行为y
                img.at<Vec3b>(j - round(ystart3) + 1, i - round(start3) + 1)[1] = srctemp.at<Vec3b>(j2, i2)[1];
                img.at<Vec3b>(j - round(ystart3) + 1, i - round(start3) + 1)[2] = srctemp.at<Vec3b>(j2, i2)[2];
                //img.at<Vec3b>(j-round(start2)+1,i-round(start1))[0]=srctemp.at<Vec3b>(j2,i2)[0];
                //img.at<Vec3b>(j-round(start2)+1,i-round(start1))[1]=srctemp.at<Vec3b>(j2,i2)[1];
                //img.at<Vec3b>(j-round(start2)+1,i-round(start1))[2]=srctemp.at<Vec3b>(j2,i2)[2];
            }
        }
        cout << "!5" << endl;


       // for(auto iter:rotationlines) {
       //
       //     line(img, cv::Point(iter[0] - round(ystart3) + 1, iter[1] - round(start3) + 1), cv::Point(iter[2]- round(ystart3) + 1, iter[3]- round(start3) + 1), CV_RGB(255,0,0),
       //          0.5);//先在图上划了线再对应过去
       // }


        //resize(img,img,cv::Size(640,round(640/ceres::abs(round(end1)-round(start3))*ceres::abs(round(yend3)-round(ystart3)))));
        //resize(img,img,cv::Size(2000,round(2000/ceres::abs(round(end1)-round(start3))*ceres::abs(round(yend3)-round(ystart3)))));
        //cv::Mat img1;
        //pyrDown(img, img, Size(img.cols/2, img.rows/2));

        //scaleIntervalSampling(img,img,2,2);
        //scaleIntervalSampling(img,img,2,2);
        //scaleIntervalSampling(img,img,2,2);
        resize(img, img, cv::Size(1200, 1080), CV_INTER_AREA);
        cout << "!6" << endl;
        cout << "width=" << img.cols << endl;
        cout << "height=" << img.rows << endl;
        //namedWindow("homography",WINDOW_AUTOSIZE);
        imshow("homography", img);

        //写视频
        writer1 << img;
/////////////////////////////////////////////////////////////////////////////////////需要注释
//        imwrite("homography.jpg",img);

        img.release();
        cout << "!7" << endl;
        //waitKey(0);

    }




    //for(auto clu:clustery) {
    //    std::cout << clu << std::endl;
    //}
    //int clustercenternum=5;
    //std::vector<double> clusterx;
    //for(int i=1;i<5;i++)
    //{
    //    clusterx.push_back(i);
    //}

    //arma::mat data(clustery);
    //data=data.t();
    //std::cout<<"data ="<<data<<std::endl;

    //size_t clusters=2;
    //arma::Row<size_t> assignments;
    //arma::mat centroids;

    //KMeans<> k;
    //k.Cluster()
    //k.Cluster(data,clusters,assignments,centroids);
    //std::cout<<"中心="<<centroids<<std::endl;


   // for()
   // {

    //}





    //else{
    //    vps=VPpre1;

    //}


       //////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
}





//int main(int argc, char **argv)
void LSDVP(Mat &src1, Rect rect, Mat &output, Point2f &VP)
{
    //if (argc < 2 || argc > 2)
    //{
      //  std::cout << "Usage: lsd_opencv_example imageName" << std::endl;
        //return -1;
    //}
    //cv::Mat src = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat src=src1(rect).clone();
    cv::Mat srctmp, src_gray, imagegray;
    //cv::Point2f VP;
    //std::cout<<"11111111111111"<<std::endl;
//   cv::Mat mask = cv::imread(argv[2],CV_LOAD_IMAGE_COLOR);
   // cv::Mat image;
    
    //cv::cvtColor(src, tmp, CV_RGB2GRAY);
    //tmp.convertTo(src_gray, CV_64FC1);
    //tmp.convertTo(src gray, CV_64FC1);
    
    
    //int cols  = src_gray.cols;
    //int rows = src_gray.rows;
    //image_double image = new_image_double(cols, rows);
//    std::cout<<"222222222222222"<<std::endl;
//    cv::cvtColor(mask,mask,CV_RGB2GRAY);
//    int dimensions=mask.channels();
//    std::cout<<"dimensions="<<dimensions<<std::endl;
//    int cols =mask.cols;
//    int rows =mask.rows;
    
    //cv::imshow("mask",mask);
    //cv::waitKey(0);
    
    /*for(int i=0; i<rows ; i++)
    {
      for(int j=0; j<cols ; j++)
      {
	if(mask.at<uchar>(i,j)>155)
	{
	  mask.at<uchar>(i,j)=1;
	}
	else
	{
	  mask.at<uchar>(i,j)=0;
	}
      }
    }
    
    
    for(int i=0; i<rows ; i++)
    {
      for(int j=0; j<cols ; j++)
      {
	src.at<Vec3b>(i,j)[0]=src.at<Vec3b>(i,j)[0]*mask.at<uchar>(i,j);
        src.at<Vec3b>(i,j)[1]=src.at<Vec3b>(i,j)[1]*mask.at<uchar>(i,j);
        src.at<Vec3b>(i,j)[2]=src.at<Vec3b>(i,j)[2]*mask.at<uchar>(i,j);
      }
    }*/
    
    //imshow("src",src);
    //cv::waitKey(0);
    
   cv::cvtColor(src, srctmp, CV_RGB2GRAY);
   srctmp.convertTo(src_gray, CV_64FC1);
   //cv::cvtColor(image,imagegray,CV_RGB2GRAY);
//   std::cout<<"!!!!!!!!!!!!!!!!!!"<<std::endl;
   int cols  = src_gray.cols;
   int rows = src_gray.rows;
   
   image_double image1 = new_image_double(cols, rows);
   image1->data = src_gray.ptr<double>(0);
    //image->data = src_gray.ptr<double>(0);
    ntuple_list ntl = lsd(image1);//先处理再叠加深度学习mask

    cv::Mat lsd = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Point pt1, pt2;
    for (int j = 0; j != ntl->size ; ++j)
    {
        pt1.x = int(ntl->values[0 + j * ntl->dim]);
        pt1.y = int(ntl->values[1 + j * ntl->dim]);
        pt2.x = int(ntl->values[2 + j * ntl->dim]);
        pt2.y = int(ntl->values[3 + j * ntl->dim]);
        int width = int(ntl->values[4 + j * ntl->dim]);
        cv::line(lsd, pt1, pt2, cv::Scalar(255), width, CV_AA);//在lsd全黑图上面画处理结果
    }
    free_ntuple_list(ntl);

    
    
/*     for(int i=0; i<rows ; i++)
    {
      for(int j=0; j<cols ; j++)
      {
	if(mask.at<uchar>(i,j)>155)
	{
	  mask.at<uchar>(i,j)=1;
	}
	else
	{
	  mask.at<uchar>(i,j)=0;
	}
      }
    }
*/    
    
    //cv::namedWindow("lsd1", CV_WINDOW_AUTOSIZE);
    //cv::imshow("lsd1", lsd);
    //cv::waitKey(10);
//    int height=lsd.rows;
//    int width=lsd.cols;
//    Mat lsddeeplearning=lsd.clone();
//    std::cout<<"height*width="<<height<<"*"<<width<<std::endl;
/*    for(int i=0; i<rows ; i++)
    {
      for(int j=0; j<cols ; j++)
      {
	//src.at<Vec3b>(i,j)[0]=src.at<Vec3b>(i,j)[0]*mask.at<uchar>(i,j);
        //src.at<Vec3b>(i,j)[1]=src.at<Vec3b>(i,j)[1]*mask.at<uchar>(i,j);
        //src.at<Vec3b>(i,j)[2]=src.at<Vec3b>(i,j)[2]*mask.at<uchar>(i,j);
	
	
	//下面这一句话决定是否采用了深度学习分割
	//lsd.at<uchar>(i,j)=lsd.at<uchar>(i,j)*mask.at<uchar>(i,j);
	
	//lsd.at<uchar>(i+1,j+1)=lsd.at<uchar>(i,j)*mask.at<uchar>(i+1,j+1);
        //lsd.at<Vec3b>(i,j)[1]=lsd.at<Vec3b>(i,j)[1]*mask.at<uchar>(i,j);
        //lsd.at<Vec3b>(i,j)[2]=lsd.at<Vec3b>(i,j)[2]*mask.at<uchar>(i,j);
	
        lsddeeplearning.at<uchar>(i,j)=lsddeeplearning.at<uchar>(i,j)*mask.at<uchar>(i,j);
	
	//lsddeeplearning.at<uchar>(i+1,j+1)=lsddeeplearning.at<uchar>(i,j)*mask.at<uchar>(i+1,j+1);
        //lsddeeplearning.at<Vec3b>(i,j)[1]=lsddeeplearning.at<Vec3b>(i,j)[1]*mask.at<uchar>(i,j);
        //lsddeeplearning.at<Vec3b>(i,j)[2]=lsddeeplearning.at<Vec3b>(i,j)[2]*mask.at<uchar>(i,j);
	
	
      }
    }
    
*/    
    //char* p=std::strstr(argv[1],"00");
    //std::stringstream
//    string s(argv[1]),s1,s2;
    //,s3;
    //std::cout<<"s="<<s<<std::endl;
    //cv::namedWindow("src", CV_WINDOW_AUTOSIZE);
    //cv::imshow("src", src);
//    s1=s.substr(0,6);
//    s2=s1+".jpg";
  //  s3=s1+"dl.jpg";
  //  std::cout<<"subs="<<s1<<std::endl;
  //  std::cout<<"finalstring="<<s2<<std::endl;
    
    double alpha=1;
    double beta=0.5;
    Mat dstimage[3];
    //,dstimagedl[3];
    Mat mv[3];
    //,mvdl[3];
    Mat processed;
    //,processeddl;
    split(src,mv);
    //split(src,mvdl);
    addWeighted(mv[0],alpha,lsd,beta,0.0,dstimage[0]);
    addWeighted(mv[1],alpha,lsd,beta,0.0,dstimage[1]);
    addWeighted(mv[2],alpha,lsd,beta,0.0,dstimage[2]);
    
    //addWeighted(mvdl[0],alpha,lsddeeplearning,beta,0.0,dstimagedl[0]);
    //addWeighted(mvdl[1],alpha,lsddeeplearning,beta,0.0,dstimagedl[1]);
    //addWeighted(mvdl[2],alpha,lsddeeplearning,beta,0.0,dstimagedl[2]);
    
    merge(dstimage,3,processed);
    //merge(dstimagedl,3,processeddl);
   /* imshow("LSD",processed);*/
    //imshow("mixdl",processeddl);
    //cv::imwrite("mix.jpg",processed);
    //cv::imwrite("mixdl.jpg",processeddl);
    //cv::namedWindow("lsd", CV_WINDOW_AUTOSIZE);
    //cv::namedWindow("lsddeeplearning", CV_WINDOW_AUTOSIZE);
    //std::cout<<"!!!!!"<<std::endl;
    //cv::imshow("lsd", lsd);
    
    //cv::imwrite(s2,lsd);
    //cv::imshow("lsddeeplearning", lsddeeplearning);
    //cv::imwrite(s3,lsddeeplearning);
    
    int mode = MODE_NIETO;
    int numVps = 1;
   // bool playMode = true;
   // bool stillImage = false; 
    bool verbose = false;
    cv::Size procSize = cv::Size(cols, rows);
    MSAC msac;
    msac.init(mode, procSize, verbose);
    
    Mat imgGRAY,outputImg;
   // std::cout<<"channels="<<lsd.channels()<<std::endl;
    //cv::cvtColor(lsd, imgGRAY, CV_BGR2GRAY);	//lsd是黑白图
    //cv::cvtColor(lsd, outputImg, CV_GRAY2BGR);
    imgGRAY=lsd.clone();
    cv::cvtColor(lsd, outputImg, CV_GRAY2BGR);//黑白图

   // imshow("lsd",lsd);
    
    //HoughTransformation
    /*Mat midImage,dstImage;
    Canny(src,midImage,50,250,3);
    cvtColor(midImage,dstImage,CV_GRAY2BGR);
    vector<Vec2f> lines;
    HoughLines(midImage,lines,1,CV_PI/180,350,0,0);
    for(size_t i=0;i<lines.size();i++)
    {
      float rho=lines[i][0],theta=lines[i][1];
      Point pt1,pt2;
      double a=cos(theta),b=sin(theta);
      double x0=a*rho,y0=b*rho;
      pt1.x=cvRound(x0+1000*(-b));
      pt1.y=cvRound(y0+1000*(a));
      pt2.x=cvRound(x0-1000*(-b));
      pt2.y=cvRound(y0-1000*(a));
      line(dstImage,pt1,pt2,Scalar(55,100,195),1,LINE_AA);
    }
    imshow("边缘检测图",midImage);
    imshow("效果图",dstImage);
    */
    //waitKey(0);
    
    //PHoughTransformation
  /*  Mat pmidImage,pdstImage;
    Canny(src,pmidImage,50,250,3);
    cvtColor(pmidImage,pdstImage,CV_GRAY2BGR);
    vector<Vec4i> plines;
    HoughLinesP(pmidImage,plines,1,CV_PI/180,80,50,10);
    for(size_t i=0;i<lines.size();i++)
    {
      Vec4i l=plines[i];
      line(pdstImage,Point(l[0],l[1]),Point(l[2],l[3]),Scalar(186,88,25),1,CV_AA);
    }
    imshow("Hp边缘检测图",pmidImage);
    imshow("Hp效果图",pdstImage);
    //waitKey(0);
    
    
    //imshow("srctmp",srctmp);
    //imshow("src",src);
    imshow("outputImg",outputImg);
    imshow("imgGRAY",imgGRAY);
    */
    processImage(msac, numVps, imgGRAY, outputImg, src, VP);//正常检测
    VP=VP+cv::Point2f(rect.x,rect.y);
//    processImage(msac, numVps, srctmp, src, src, VP);//hough检测
    
//    imshow("VP",outputImg);  
//    imshow("VPsrc",src); 
    output=src;
   // std::cout<<"VP.x= "<<VP.x<<"VP.y= "<<VP.y<<std::endl;
  /*  cv::waitKey(0);*/
    //cv::destroyAllWindows();
//    return 0;
}