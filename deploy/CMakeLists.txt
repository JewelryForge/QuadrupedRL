cmake_minimum_required(VERSION 3.20)

project(quadruped_deploy)

set(CONDA_ENV_PATH /home/jewel/libraries/miniconda3/envs/rl)
set(PKG_PATH /home/jewel/Workspaces/teacher-student/deploy)

set(CMAKE_PREFIX_PATH ${CONDA_ENV_PATH}/lib/python3.9/site-packages/torch/share/cmake)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(EXTRA_LIBS ${PKG_PATH}/thirdparty/unitree_legged_sdk/lib/libunitree_legged_sdk_amd64.so)

message(STATUS "TORCH_CXX_FLAGS ${TORCH_CXX_FLAGS}")

#find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

include_directories(include)
include_directories(thirdparty/unitree_legged_sdk/include)
include_directories("/usr/local/include/eigen3")

#message(STATUS "${EIGEN3_INCLUDE_DIRS}")
add_executable(alienGo_policy src/main.cpp)
target_link_libraries(alienGo_policy ${TORCH_LIBRARIES} ${EXTRA_LIBS} lcm)

set_property(TARGET alienGo_policy PROPERTY CXX_STANDARD 14)
