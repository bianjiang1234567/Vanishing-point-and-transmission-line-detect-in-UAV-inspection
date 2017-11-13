#ifndef LM_
#define LM_
#include "ceres/ceres.h"
//#include "glog/logging.h"
#include "Eigen/Dense"
//#include <boost/algorithm/string/split.hpp>
//#include <boost/algorithm/string/classification.hpp>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
using namespace cv;
using namespace std;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::CauchyLoss;
//using ceres::TolerantLoss;
// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
typedef Eigen::Matrix<double,4,1> Line;
typedef Eigen::Matrix<double,2,1> LineMid;
typedef Eigen::Matrix<double,3,1> LineEq;
struct VPCostFunctor {
    VPCostFunctor(double midx, double midy, double endx, double endy) : _midx(midx), _midy(midy), _endx(endx), _endy(endy) {}

    template <typename T> bool operator()(const T* const vp_x, const T* const vp_y, T* residual) const {//vpx vpy 是优化变量
    T a,b,c;
    a=T(_midy)-vp_y[0];
    b=vp_x[0]-T(_midx);
    c=T(-_midy)*vp_x[0]+T(_midx)*vp_y[0];
    T numerator = a*T(_endx)+b*T(_endy)+c;
    numerator=numerator*numerator;
    T denominator = a*a+b*b;
    //Eigen::Matrix<double,3,1> homo_vp(vp_x[0],vp_y[0],1);
    //Eigen::Vector3d homo_mid(_midx,_midy,1);
    //LineEq line_mid2vp(homo_vp.cross(homo_mid));
    //double numerator = line_mid2vp.dot(homo_mid);
    //numerator = numerator*numerator;
    //double denominator= line_mid2vp[0]*line_mid2vp[0]+line_mid2vp[1]*line_mid2vp[1];
    //residual[0] =T(numerator/denominator);
    residual[0]=numerator/denominator;
    return true;
  }

private:
    const double _midx;
    const double _midy;
    const double _endx;
    const double _endy;
};
void lm(vector<Vec4i>& lines, Point2f& vp);
#endif
