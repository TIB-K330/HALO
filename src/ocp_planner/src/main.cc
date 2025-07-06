#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
// #include <nlohmann/json.hpp>

#include <ros/ros.h>

#include "backward.hpp"
#include "ocp_planner/ControlVar.h"
#include "ocp_planner/JointState.h"
#include "ocp_planner/OcpLocalPlann.h"
#include "planner.h"

// using json = nlohmann::json;

namespace backward {
backward::SignalHandling sh;
}

namespace robot_plann {
//
std::unique_ptr<robot_plann::Planner> _planner;
int _verbose = 1;
ros::Publisher astar_path_pub;


bool PlannSrvCallback(ocp_planner::OcpLocalPlann::Request &req,
                      ocp_planner::OcpLocalPlann::Response &res) {
  robot_plann::JointState ob_state;
  const auto &robot_state = req.ob.robot_state;
  ob_state.robot.px = robot_state.pose.x;
  ob_state.robot.py = robot_state.pose.y;
  ob_state.robot.yaw = robot_state.pose.theta;
  ob_state.robot.v = 0.5 * (robot_state.vr + robot_state.vl);
  ob_state.robot.yaw_rate =
      0.5 * (robot_state.vr - robot_state.vl) / robot_state.radius;

  ob_state.robot.v_pref = robot_state.v_pref;
  ob_state.robot.radius = robot_state.radius;
  ob_state.robot.gx = robot_state.gx;
  ob_state.robot.gy = robot_state.gy;

  for (const auto &hum_iter : req.ob.human_states) {
    robot_plann::HumanState hum_state;
    hum_state.px = hum_iter.px;
    hum_state.py = hum_iter.py;
    hum_state.vx = hum_iter.vx;
    hum_state.vy = hum_iter.vy;
    hum_state.radius = hum_iter.radius;
    ob_state.hum.push_back(hum_state);
  }

  for (const auto &obst_iter : req.ob.obstacle_states) {
    robot_plann::ObstacleState obst_state;
    obst_state.px = obst_iter.px;
    obst_state.py = obst_iter.py;
    obst_state.radius = obst_iter.radius;
    ob_state.obst.push_back(obst_state);
  }

  // clock-wise
  ob_state.rect.vertices.clear();
  for (const auto &poly_iter : req.ob.poly_states) {
    for (const auto &vertex : poly_iter.vertices) {
      Eigen::Vector2d pt(vertex.x, vertex.y);
      ob_state.rect.vertices.push_back(pt);
    }
    break;
  }

  ob_state.walls.clear();
  for (const auto &input_wall : req.ob.walls) {
    Wall wall = Planner::ClipWall(input_wall.sx, 
                                  input_wall.sy, 
                                  input_wall.ex, 
                                  input_wall.ey, -5.5, 5.5);

    ob_state.walls.push_back(wall);
  }

  Eigen::Vector2d sub_goal = {req.sub_goal.x, req.sub_goal.y};

  auto start_stamp = std::chrono::high_resolution_clock::now();
  auto mpc_return = _planner->PlannExec(ob_state, sub_goal);
  auto end_stamp = std::chrono::high_resolution_clock::now();
  double time_cost =
      std::chrono::duration<double, std::milli>(end_stamp - start_stamp)
          .count();
  

  res.astar_path.clear();

  std::vector<robot_plann::Point> astar_path = _planner->GetAStarPath();
  for (int i = 0; i < astar_path.size(); ++i) {
    ocp_planner::Point pt;
    pt.x = astar_path.at(i).x;
    pt.y = astar_path.at(i).y;
    res.astar_path.push_back(pt);

  }

  res.control_vars.clear();
  ocp_planner::ControlVar cur_control_var{};
  res.success = false;
  if (!mpc_return.success) {
    // res.al = -req.ob.robot_state.vl / (kDT * 0.5);
    // res.ar = -req.ob.robot_state.vr / (kDT * 0.5);
    res.al = -req.ob.robot_state.vl / kDT;
    res.ar = -req.ob.robot_state.vr / kDT;

    res.revised_goal.x = sub_goal.x();
    res.revised_goal.y = sub_goal.y();
    if (_verbose >= 1) std::cout << "Ocp plann failed!" << std::endl;
    cur_control_var.al = res.al;
    cur_control_var.ar = res.ar;
    for (int i = 0; i < kNP; ++i) {
      res.control_vars.push_back(cur_control_var);
    }

  } else {
    res.al = mpc_return.stages[0].uk.acc - mpc_return.stages[0].uk.dr * 0.3;
    res.ar = mpc_return.stages[0].uk.acc + mpc_return.stages[0].uk.dr * 0.3;
    res.revised_goal.x = sub_goal.x();
    res.revised_goal.y = sub_goal.y();

    for (int i = 0; i < kNP; ++i) {
      cur_control_var.al =
          mpc_return.stages.at(i).uk.acc - mpc_return.stages.at(i).uk.dr * 0.3;
      cur_control_var.ar =
          mpc_return.stages.at(i).uk.acc + mpc_return.stages.at(i).uk.dr * 0.3;
      res.control_vars.push_back(cur_control_var);
    }
    res.success = true;
    if (_verbose >= 1) std::cout << "Time cost: " << time_cost << std::endl;
  }

   nav_msgs::Path refline_line = _planner->GetAStarSmoothPath();
   refline_line.header.stamp = ros::Time::now();
   if (!refline_line.poses.empty()) {
     astar_path_pub.publish(refline_line);
   }


  return true;
}

void TestRun() {}

}  // namespace robot_plann

int main(int argc, char **argv) {
  ros::init(argc, argv, "opt_planner");
  ros::NodeHandle nh;

  robot_plann::_planner =
      std::make_unique<robot_plann::Planner>(robot_plann::_verbose);


  auto plann_srv =
      nh.advertiseService("/ocp_plann", robot_plann::PlannSrvCallback);

  robot_plann::astar_path_pub = nh.advertise<nav_msgs::Path>("/a_star_path", 1);

  ros::spin();
  return 0;
}
