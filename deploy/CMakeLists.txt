cmake_minimum_required(VERSION 3.20)

project(alienGo_deploy)

set(LIBTORCH_PATH $ENV{HOME}/libraries/libtorch)
set(PKG_PATH $ENV{HOME}/Workspaces/teacher-student/deploy)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${LIBTORCH_PATH}/share/cmake)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(EXTRA_LIBS ${PKG_PATH}/thirdparty/unitree_legged_sdk/lib/libunitree_legged_sdk_amd64.so)

find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)
find_package(catkin REQUIRED COMPONENTS
        roscpp std_msgs nav_msgs tf message_generation)

add_message_files(FILES FloatArray.msg MultiFloatArray.msg)
generate_messages()
catkin_package(CATKIN_DEPENDS roscpp std_msgs nav_msgs tf message_runtime)

include_directories(include)
include_directories(${catkin_INCLUDE_DIRS})
include_directories(thirdparty/unitree_legged_sdk/include)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(alienGo_policy src/main.cpp)
target_link_libraries(alienGo_policy ${catkin_LIBRARIES} ${TORCH_LIBRARIES} ${EXTRA_LIBS} lcm)

set_property(TARGET alienGo_policy PROPERTY CXX_STANDARD 14)
