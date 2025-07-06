/**
  ******************************************************************************
  * @file    planner.h
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2023/05/05
  * @brief   ocp planner as rl backend for MobiRo @ tib_k331
  ******************************************************************************
  * @attention
  *
  ******************************************************************************
  */
#ifndef PLANNER_H
#define PLANNER_H
#include <chrono>
#include <iostream>

#include "types.h"
#include "lookahead.h"

#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include "mpc.h"


namespace robot_plann {

inline cv::Point MapCoord2ImgIdx(const Eigen::Vector2d &pt, bool vis = false) {
  if (abs(pt(0)) > kHalfMapWidth || abs(pt(1)) > kHalfMapHeight) {
    std::stringstream err;
    err << "Invalid coord: " << pt(0) << ", " << pt(1);
    throw err.str();
  }
  cv::Point idx;
  idx.x = (int)round((pt(0) + kHalfMapWidth) / kMapResol);
  idx.y = (int)round((kHalfMapHeight - pt(1)) / kMapResol);
  if (vis) {
    idx.x = (int)round(idx.x * kVisualScale);
    idx.y = (int)round(idx.y * kVisualScale);
  }
  return idx;
}

inline Eigen::Vector2d ImgIdx2MapCoord(const cv::Point &idx,
                                       bool vis = false) {
  Eigen::Vector2d pt;
  if (vis) {
    pt(0) = idx.x / kVisualScale * kMapResol - kHalfMapWidth;
    pt(1) = kHalfMapHeight - idx.y / kVisualScale * kMapResol;
  } else {
    pt(0) = idx.x * kMapResol - kHalfMapWidth;
    pt(1) = kHalfMapHeight - idx.y * kMapResol;
  }
  return pt;
}

inline void Vertices2Contour(const std::vector<Eigen::Vector2d> &vertices,
                             std::vector<cv::Point> &contour) {
  if (contour.size()) contour.clear();
  for (int i = 0; i < (int)vertices.size(); i++) {
    contour.push_back(MapCoord2ImgIdx(vertices[i]));
  }
}

class Planner {
 public:
  explicit Planner(int verbose = 0)
    : has_map_(false), 
      visual_flag_(false),
      move_forward_(true),
      verbose_(verbose) {
    astar_planner_ = std::make_unique<AStar>(verbose);

    path_smoother_ = std::make_unique<SmoothCorner>();
    path_smoother_->SetDeltaMax(0.1);

    vel_planner_ = std::make_unique<LookAhead>();
    vel_planner_->SetParams(kMaxLinearVel, kMaxLinearAcc, kMaxAngularVel);
    ocp_planner_ = std::make_unique<Mpc>(verbose);

    MpcParams::Ptr mpc_params = std::make_shared<MpcParams>();
    mpc_params->dt = kDT;
    mpc_params->np = kNP;
    mpc_params->max_linear_vel  = kMaxLinearVel;
    mpc_params->max_linear_acc  = kMaxLinearAcc;
    mpc_params->max_angular_vel = kMaxAngularVel;
    mpc_params->max_angular_acc = kMaxAngularAcc;
    mpc_params->local_obst_num = 8;
    ocp_planner_->SetParams(mpc_params);
    a_start_smooth_path_ = std::make_shared<nav_msgs::Path>();
    // update map
    cv::Mat map = this->CreateMap();
    this->SetMap(map);
    
  }

  void SetMap(const cv::Mat &map) {
    map_ = map;
    astar_planner_->InitMap(map_);
    has_map_ = true;
  }

  MpcReturn PlannExec(const JointState &state, Eigen::Vector2d &sub_goal);

  robot_plann::MPCOutputForPython RunSlover(const robot_plann::MPCInputForPython& input);


  bool visual_flag_;

  inline nav_msgs::Path GetAStarSmoothPath() const {
    return *a_start_smooth_path_;
  }

  inline std::vector<Point> GetAStarPath() const {
    return astar_path_;
  }

  static Wall ClipWall(double x1, double y1, double x2, double y2, 
                      double x_min = -5, double x_max = 5);

  
 private:
  void UpdateCostMap(const JointState &state);
  cv::Mat CreateMap(); 
  bool CheckAround(Eigen::Vector2d &pos);
  bool CheckNavGoal(Eigen::Vector2d &sub_goal,
                    const Eigen::Vector2d &nav_goal, 
                    const Eigen::Vector2d &pos, 
                    const std::vector<Eigen::Vector2d> &vertices);
  
  bool SimpleRayCast(Eigen::Vector2d &goal, const Eigen::Vector2d &pos);
  Eigen::Vector2d PidCalc(const JointState &state);
  static void PybindInputDataChange(const robot_plann::MPCInputForPython& input,
                                    robot_plann::JointState &ob_state);

  // planner sub-modules
  std::unique_ptr<AStar> astar_planner_;

  std::unique_ptr<SmoothCorner> path_smoother_;
  std::unique_ptr<LookAhead> vel_planner_;

  std::unique_ptr<Mpc> ocp_planner_;

  std::shared_ptr<nav_msgs::Path> a_start_smooth_path_;
  std::vector<Point> astar_path_{};

  cv::Mat map_, cost_map_, visual_map_;
  bool has_map_;
  bool move_forward_;
  int verbose_;

};

}  // namespace robot_plann

# endif  // PLANNER_H