cmake_minimum_required(VERSION 2.8.3)
project(additional_view_registration_server)

set(CMAKE_CXX_FLAGS "-O4 -fPIC -std=c++0x -fpermissive ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS roscpp rospy siftgpu qt_build)

catkin_package(
   INCLUDE_DIRS 
   CATKIN_DEPENDS siftgpu
   DEPENDS libceres-dev suitesparse qt_build libqt4-dev
   LIBRARIES additional_view_registration_optimizer
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/home/thomas/Projects/perception_stack_apa/3rdparty/SiftGPU/src/siftgpu_external/src")
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

rosbuild_prepare_qt4(QtCore QtXml)

FIND_PACKAGE(Ceres REQUIRED)
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})

include_directories(${catkin_INCLUDE_DIRS}
                    include)


add_library(additional_view_registration_optimizer include/additional_view_registration_server/additional_view_registration_optimizer.h
                                               include/additional_view_registration_server/additional_view_registration_residual.h
                                               include/additional_view_registration_server/sift_wrapper.h
                                               src/additional_view_registration_optimizer.cpp
                                               src/sift_wrapper.cpp)                                                                                      

target_link_libraries(additional_view_registration_optimizer ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${QT_LIBRARIES} ${CERES_LIBRARIES})

add_executable(test_registration src/test_registration.cpp)

target_link_libraries(test_registration
    additional_view_registration_optimizer ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${QT_LIBRARIES} ${CERES_LIBRARIES}
)

