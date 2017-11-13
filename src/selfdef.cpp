#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include "selfdef.h"
//#include "selfdef.h"
//using namespace std;
//using namespace cv;
void FilterHorizonLine(
         vector<Vec4i> & lines, int width, int height
        ){
        cout<<"*************************************"<<endl;
        vector<Vec4i> templines=lines;
        lines.clear();
        for(size_t i=0;i<templines.size();i++){
        Vec2f line_vector((templines[i][0]-templines[i][2]),(templines[i][1]-templines[i][3]));
	//float normnum=sqrt(pow(line_vector.x,2)+pow(line_vector.y,2));
	float normnum=norm(line_vector,NORM_L2);

	
	//if(0.5*(templines[i][0]+templines[i][2])<0.5*1280 || 0.5*(templines[i][1]+templines[i][3])>0.5*720)
    //        if(0.5*(templines[i][0]+templines[i][2])<0.85*1280 || 0.5*(templines[i][1]+templines[i][3])>0.5305*720)
    //        if(0.5*(templines[i][0]+templines[i][2])<0.85*1280 || 0.5*(templines[i][1]+templines[i][3])>0.518*720)
            //if(0.5*(templines[i][0]+templines[i][2])<0.85*1280 || 0.5*(templines[i][1]+templines[i][3])>375)
            //if(0.5*(templines[i][0]+templines[i][2])<0.85*1280 || 0.5*(templines[i][1]+templines[i][3])>375)
            if(0.5*(templines[i][0]+templines[i][2])<0.85*1280 || templines[i][3]>381)
            //     if(0.5*(templines[i][1]+templines[i][3])>375)
	{
	  continue;
	}

	if(normnum<100)
	{
	  continue;
	}

    //        if(normnum>200)
    //        {
    //            continue;
    //        }

	line_vector=normalize(line_vector);
	float cos_theta=line_vector.dot(Vec2f(1.0,0.0));

        //if(abs(cos_theta)>0.95||abs(cos_theta)<0.15){
	if(abs(cos_theta)<0.15){
            //cout<<"cos_theta =  "<<cos_theta<<endl;
            continue;
        }
            //std::cout<<"norm= "<<normnum<<std::endl;
            //cout<<"cos_theta  ="<<endl<<cos_theta<<endl;
        lines.push_back(templines[i]);
            if(lines.size()==0)
            {
                lines=templines;
            }
            //cout<<"lines size  ="<<lines.size()<<endl;
        }
}
