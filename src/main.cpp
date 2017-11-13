#include "LSDVP.h"
using namespace cv;
using namespace std;
int flag1=0;
int frame=0;
int main()
{
    cv::VideoWriter writer("finalshow.avi",0,20.0,Size(1280,720),true);
    cv::VideoCapture capture;//视频
    //capture.open("final11.avi");//视频
    //capture.open("1.mov");//视频
    capture.open("final1.avi");//视频

    cv::Point2f VPpre;
    int count=0;
    while(1) {
        count++;
        frame++;
        cv::Point2f VP(0,0);
/////////////////////////////////////////////////////////////////////////////////////需要注释
//       cv::Mat src = cv::imread("./cluster/005912.bmp", CV_LOAD_IMAGE_COLOR), output;

        cv::Mat src;//视频
        cv::Mat output;//视频
        capture>>src;//视频

        if(src.empty())
        {
            break;
        }

        //resize(src,src,Size(1280,720));
        //imwrite("0000011280.jpg",src);
        //int x=0,y=0,height=720;
        //int width=1280-x;
        //cv::resize(src, src, Size(1500, 1500));
        cv::resize(src,src,Size(1280,720));

        int x = 0, y = 0, height =720;
        int width = 1280 - x;

        //int x = 0, y = 0, height =720;
        //int width = 1280 - x;

        Rect rect(x, y, width, height);

        LSDVP(src, rect, output, VP);
        //if(count==1 && flag1==0) {


        if( VP.x!=0 && VP.y!=0 && flag1==0 && VP.x>0 && VP.x<1280 && VP.y>0 && VP.y<720) { //VP.x!=0 && VP.y!=0 是检测到了   VP.x>0 && VP.x<1280 && VP.y>0 && VP.y<720 只要检测到好点就记录并且更新 消失点就不会再跑出那个小圈子
            VPpre = VP;
            flag1=1;//不再进去
        }
        if((VP.x==rect.x && VP.y==rect.y) || abs(VP.x-VPpre.x)>200 || abs(VP.x-VPpre.y)>200)//VP没有算出来的话会是（0,0)  LSDVP内部函数会跳过VP计算部分
        {//突变超过200 认为不好的点 少于200 可以被慢慢带过去
            cout<<"!!!!NO VP"<<endl;
            VP=VPpre;
        }
        else
        {
            VPpre=VP;
        }
        std::cout << "VP.x= " << VP.x << " VP.y= " << VP.y << std::endl;

        //rectangle(output, rect, Scalar(0, 255, 0));//绿色的框

        imshow("result", output);
        writer<<output;

//////////////////////////////////////////////////////////////////////////////////////////需要注释
//        imwrite("result.jpg",output);
        waitKey(1);
        circle(src, VP, 6, CV_RGB(0, 0, 255), 2);//画在result上 若要显示红点 roi区域增大
        circle(src, VP, 5, CV_RGB(0, 0, 255), -1);
        imshow("src", src);
        waitKey(1);

        //std::cout<<"count"<<std::endl;
    }
    std::cout<<"final"<<std::endl;
  return 0;
}
