#include <cv.h>
#include <lsd.h>
#include <opencv2/core/core.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <string>

#define USE_PPHT
#define MAX_NUM_LINES	200

#include "MSAC.h"
#include "selfdef.h"

void LSDVP(Mat &src, Rect rect, Mat &output, Point2f &VP);
void processImage(MSAC &msac, int numVps, cv::Mat &imgGRAY, cv::Mat &outputImg, cv::Mat &srctemp, cv::Point2f &VPET);
