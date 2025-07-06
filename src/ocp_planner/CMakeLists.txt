cmake_minimum_required(VERSION 3.0.2)
project(ocp_planner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  message_generation
)


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  DynaObstacle.msg
  ObstacleArray.msg
  RobotState.msg
  HumanState.msg
  ObstacleState.msg
  WallState.msg
  Point.msg 
  PolyState.msg
  JointState.msg
  ControlVar.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  OcpLocalPlann.srv 
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ocp_planner
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME}_node
  src/main.cc
  src/astar.cc
  src/smooth.cc
  src/lookahead.cc
  src/planner.cc
  src/mpc.cc

)

target_link_libraries(${PROJECT_NAME}_node 
  # PRIVATE 
  # ${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  casadi
  dw
)



# catkin_install_python(PROGRAMS scripts/test.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

################################################
################# pybind11 #####################
################################################

set(PYTHON_EXECUTABLE "/home/wangchao/miniconda3/envs/mpc_rl/bin/python")
set(PYTHON_INCLUDE_DIRECTORY "/home/wangchao/miniconda3/envs/mpc_rl/include/python3.8")

add_subdirectory(extern/pybind11)

pybind11_add_module(${PROJECT_NAME}_py
  src/ocp_bind.cc
  src/astar.cc
  src/smooth.cc
  src/lookahead.cc
  src/planner.cc
  src/mpc.cc

)
target_link_libraries(${PROJECT_NAME}_py
  PRIVATE 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  casadi

)