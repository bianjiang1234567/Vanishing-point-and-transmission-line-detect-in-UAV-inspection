// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

//#include "ceres/ceres.h"
//#include "glog/logging.h"
//#include "Eigen/Dense"
//#include <boost/algorithm/string/split.hpp>
//#include <boost/algorithm/string/classification.hpp>
//#include <fstream>
//using ceres::AutoDiffCostFunction;
//using ceres::CostFunction;
//using ceres::Problem;
//using ceres::Solver;
//using ceres::Solve;

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.

//typedef Eigen::Matrix<double,4,1> Line;
//typedef Eigen::Matrix<double,2,1> LineMid;
//typedef Eigen::Matrix<double,3,1> LineEq;
//struct VPCostFunctor {
//    VPCostFunctor(double midx, double midy, double endx, double endy) : _midx(midx), _midy(midy), _endx(endx), _endy(endy) {}
//
//    template <typename T> bool operator()(const T* const vp_x, const T* const vp_y, T* residual) const {//vpx vpy 是优化变量
//    T a,b,c;
//    a=T(_midy)-vp_y[0];
//    b=vp_x[0]-T(_midx);
//    c=T(-_midy)*vp_x[0]+T(_midx)*vp_y[0];
//    T numerator = a*T(_endx)+b*T(_endy)+c;
//    numerator=numerator*numerator;
//    T denominator = a*a+b*b;

//Eigen::Matrix<double,3,1> homo_vp(vp_x[0],vp_y[0],1);
    //Eigen::Vector3d homo_mid(_midx,_midy,1);
    //LineEq line_mid2vp(homo_vp.cross(homo_mid));
    //double numerator = line_mid2vp.dot(homo_mid);
    //numerator = numerator*numerator;
    //double denominator= line_mid2vp[0]*line_mid2vp[0]+line_mid2vp[1]*line_mid2vp[1];
    //residual[0] =T(numerator/denominator);

//    residual[0]=numerator/denominator;
//    return true;
//  }

//private:
//    const double _midx;
//    const double _midy;
//    const double _endx;
//    const double _endy;
//};
#include "lm.h"
#include "chrono"

void lm(vector<Vec4i> & lines, Point2f& vp) {
  //google::InitGoogleLogging(argv[0]);
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  //chrono::steady_clock::time_point t1=chrono::steady_clock::now();
  std::vector<Line> line_list;
  std::vector<LineEq> line_eq_list;
  std::vector<LineMid> line_mid_list;
////////////////////////////////////text input for line_list///////////////////////////////////
  //std::ifstream line_data("/home/hxl/slambook-master/ch6/ceres_curve_fitting/build/data.txt");
  
  //std::ifstream line_data("/home/Downloads/vanishingPointLM/hxl/build/data.txt");
  //std::vector<std::string> str_list;
  //std::string str_item;
  //while(std::getline(line_data,str_item)){
  //   std::vector<std::string> items_line;
  //   boost::split(items_line, str_item, boost::is_any_of(","));
  //   Line line_item(std::atof(items_line[0].c_str()),std::atof(items_line[1].c_str()),std::atof(items_line[2].c_str()),std::atof(items_line[3].c_str()));
  //   line_list.push_back(line_item);
  //}
///////////////////////////////////////////////////////////////////////////////////////////////
  for(auto ktem:lines){
      //LineEq eq(Eigen::Vector3d(ktem[0],ktem[1],1).cross(Eigen::Vector3d(ktem[2],ktem[3],1)));
      LineMid line_mid(0.5*(ktem[0]+ktem[2]),0.5*(ktem[1]+ktem[3]));
      //line_eq_list.push_back(eq);
      line_mid_list.push_back(line_mid);
  }
 // Eigen::MatrixXd B(line_eq_list.size(),2);
 // Eigen::VectorXd c(line_eq_list.size());
 // int B_row_index=0;
 // for(auto ltem:line_eq_list){
//    B(B_row_index,0)=ltem[0];
 //   B(B_row_index,1)=ltem[1];
 //   c(B_row_index)=-ltem[2];
  //  B_row_index++;
 //}
 // std::cout<<"B "<<std::endl<<B<<std::endl;
 // std::cout<<"c "<<std::endl<<c<<std::endl;
 // Eigen::Vector2d invB_c=B.fullPivLu().solve(c);
 // std::cout<<"invB_c "<<std::endl<<invB_c<<std::endl;
  // Build the problem.
  //double vpx=invB_c[0]+0.15;
  //double vpy=invB_c[1]+0.15;
  double vpx=vp.x;
  double vpy=vp.y;
  Problem problem;
  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  for(int i=0;i<lines.size();i++){
  LineMid mid_ = line_mid_list[i];
  Line line_(lines[i][0],lines[i][1],lines[i][2],lines[i][3]);
  //problem.AddResidualBlock(new AutoDiffCostFunction<VPCostFunctor, 1, 1, 1>(new VPCostFunctor(mid_[0],mid_[1],line_[0],line_[1])), NULL, &vpx, &vpy);//对vpx vpy优化 mid line为参数量
    problem.AddResidualBlock(new AutoDiffCostFunction<VPCostFunctor, 1, 1, 1>(new VPCostFunctor(mid_[0],mid_[1],line_[0],line_[1])), new CauchyLoss(0.5), &vpx, &vpy);
  }

  Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  Solver::Summary summary;
  chrono::steady_clock::time_point t1=chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);
  chrono::steady_clock::time_point t2=chrono::steady_clock::now();
  chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout<<"solve time cost="<<time_used.count()<<"seconds"<<endl;
  //std::cout << summary.BriefReport() << "\n";
  //std::cout << "x : " << initial_x
            //<< " -> " << x << "\n"
  std::cout<<"lm vp "<<vpx<<" "<<vpy<<std::endl;
  vp.x=vpx;
  vp.y=vpy;
  double precision=0.0;
  //std::cout<<"precision "<<(A*y).isApprox(b,precision)<<std::endl;
  //std::cout<<precision<<std::endl;
  //return 0;
}
