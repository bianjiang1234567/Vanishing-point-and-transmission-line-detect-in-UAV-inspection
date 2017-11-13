cmake_minimum_required(VERSION 3.0)


set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(OpenCV_DIR "/usr/local/opencv-3.1.0/share/OpenCV")
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR} "/home/bianjiang/Downloads/LSDVProiellipsecluster/include")

# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

set(SOURCE_FILES 
	lsd.cpp
	MSAC.cpp
	errorNIETO.cpp
	lmmin.cpp
		selfdef.cpp
        LSDVP.cpp
        lm.cpp
		k-means.cpp
)

add_executable(main main.cpp ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS} ${CERES_LIBRARIES})

#TARGET_LINK_LIBRARIES(lsd_opencv_example lsd ${OpenCV_LIBS})
#set(lsd "/home/bianjiang/Downloads/LSD-OpenCV-MATLAB-master/build/lib/liblsd.a")
#TARGET_LINK_LIBRARIES(1 ${lsd} ${OpenCV_LIBS})
