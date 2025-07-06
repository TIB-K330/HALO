/**
  ******************************************************************************
  * @file    planner.cc
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2023/05/05
  * @brief   ocp planner as rl backend for MobiRo @ tib_k331
  ******************************************************************************
  * @attention
  *
  ******************************************************************************
  */
#include "planner.h"

namespace robot_plann {

MpcReturn Planner::PlannExec(const JointState &state,
                             Eigen::Vector2d &sub_goal) {
  if (!has_map_) return {MpcStages(), false};
  try {
    this->UpdateCostMap(state);
  } catch (const std::string &e) {
    if (verbose_ >= 1) {
      std::cout << "Cost map update failed! " << e << std::endl;
    }
    return {MpcStages(), false};
  }
  
  Eigen::Vector2d start_pt = {state.robot.px, state.robot.py};
  const Eigen::Vector2d goal_pt = {state.robot.gx, state.robot.gy};

  if (!CheckAround(start_pt)) {
    if (verbose_ >= 1) {
      std::cout << "Failed to escape from obstacle! " << std::endl;
    }
    return {MpcStages(), false};
  } 


  if (!CheckNavGoal(sub_goal, goal_pt, start_pt, state.rect.vertices)) {
    if (verbose_ >= 1) {
      std::cout << "Sub goal is unreachable!!!" << std::endl;
    }
    return {MpcStages(), false};
  }
  
  if (verbose_ > 1) std::cout << "DEBUG: check done!" << std::endl;

  auto astar_path = astar_planner_->SearchPath(cost_map_, start_pt, sub_goal);
  if (astar_path.size() < 2) {
    if (verbose_ >= 1) { 
      std::cout << "Invalid A* path with len = " << astar_path.size() << 
          std::endl;
    }
    return {MpcStages(), false};
  }

  auto smooth_path = path_smoother_->SmoothSharpCorner(cost_map_, astar_path);

  auto final_path = vel_planner_->UpdateVelocity(smooth_path, state.robot.v);

  {
    a_start_smooth_path_->header.frame_id = "map";
    a_start_smooth_path_->poses.clear();
    geometry_msgs::PoseStamped pose_stamped;
    for (const Point &p: final_path) {
      pose_stamped.pose.position.x = p.x;
      pose_stamped.pose.position.y = p.y;
      pose_stamped.pose.position.z = p.v;
      a_start_smooth_path_->poses.push_back(pose_stamped);
    }
    astar_path_.clear();
    astar_path_ = final_path;

  }


  auto revised_state = state;
  if (final_path.size() < 2) {
    if (verbose_ >= 1) {
      std::cout << "Invalid reference trajectory with len = " << 
          final_path.size() << std::endl;
    }
    Eigen::Vector2d pid_acc = this->PidCalc(state);
    auto mpc_stages = MpcStages();
    mpc_stages[0].uk.acc = pid_acc(0);
    mpc_stages[0].uk.dr = pid_acc(1);
    return {mpc_stages, true};
  } 
  else {
    double phi_0 = atan2(final_path[1].y - final_path[0].y,
                         final_path[1].x - final_path[0].x);

    double yaw_error = state.robot.yaw - phi_0;
    Unwrap(yaw_error);
    
    if (abs(yaw_error) > M_PI / 2) {
      revised_state.robot.yaw -= M_PI;
      Unwrap(revised_state.robot.yaw);
      revised_state.robot.v = -revised_state.robot.v;
      move_forward_ = false;
    } else {
      move_forward_ = true;
    }

    while(final_path.size() < kNP) {
      final_path.push_back(final_path.back());
    }
  }
  
  if (verbose_ > 1) std::cout << "DEBUG: geo plann done!" << std::endl;

  auto mpc_return = ocp_planner_->RunMpc(revised_state, final_path);

  // std::cout << "===" << std::endl;
  if (visual_flag_) {
    for (int i = 0; i < final_path.size(); i++) {
      Eigen::Vector2d path_pt(final_path[i].x, final_path[i].y);
      cv::Point visual_pt = MapCoord2ImgIdx(path_pt, true);
      cv::circle(visual_map_, visual_pt, 3, cv::Scalar(200, 0, 0), 2);
    }
    if (mpc_return.success) {
      for (int k = 0; k < mpc_return.stages.size(); k++) {
        Eigen::Vector2d path_pt(mpc_return.stages[k].xk.X, 
                                mpc_return.stages[k].xk.Y);
        
        cv::Point visual_pt = MapCoord2ImgIdx(path_pt, true);
        cv::circle(visual_map_, visual_pt, 3, cv::Scalar(0, 0, 200), 2);
      }
    }

    cv::imshow("debug!", visual_map_);
    cv::waitKey(5);
  }
  if (mpc_return.success && !move_forward_) {
    for (std::size_t t = 0; t < mpc_return.stages.size(); ++t) {
      mpc_return.stages[t].uk.acc = -mpc_return.stages[t].uk.acc;
    }
    // mpc_return.stages[0].uk.acc = -mpc_return.stages[0].uk.acc;
  }
  return mpc_return;
}

bool Planner::CheckAround(Eigen::Vector2d &pos) {
  cv::Point pixel;
  try{
    pixel = MapCoord2ImgIdx(pos);
  } catch (const std::string &e) {
    std::cout << e << std::endl;
    return false;
  }
  if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) return true;
  
  auto is_free = [this](int x, int y) {
    return x > 0 && x < cost_map_.cols && y > 0 && y < cost_map_.rows &&
           cost_map_.at<u_char>(y, x) > 240;
  };

  int x_right_free = 0, x_left_free = 0, y_up_free = 0, y_down_free = 0;
  for (int i = 0; i < 15; i++) {
    if (is_free(pixel.x + i, pixel.y)) x_right_free++;
    if (is_free(pixel.x, pixel.y - i)) y_up_free++;
    if (is_free(pixel.x - i, pixel.y)) x_left_free++;
    if (is_free(pixel.x, pixel.y + i)) y_down_free++;
  }

  if (x_right_free > std::max({y_up_free, x_left_free, y_down_free})) {
    // should escape from x right which has the least occupancy pixel counts
    while (++pixel.x < cost_map_.cols) {
      if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) {
        pos = ImgIdx2MapCoord(pixel);
        if (verbose_ >=1) std::cout << "Escape from x right!" << std::endl;
        return true;
      }
    }
  } else if (y_up_free > std::max({x_left_free, y_down_free})) {
    while (--pixel.y > 0) {
      if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) {
        pos = ImgIdx2MapCoord(pixel);
        if (verbose_ >=1) std::cout << "Escape from y up!" << std::endl;
        return true;
      }      
    }
  } else if (x_left_free > y_down_free) {
    while (--pixel.x > 0) {
      if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) {
        pos = ImgIdx2MapCoord(pixel);
        if (verbose_ >=1) std::cout << "Escape from x left!" << std::endl;
        return true;
      }      
    }
  } else {
    while (++pixel.y < cost_map_.rows) {
      if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) {
        pos = ImgIdx2MapCoord(pixel);
        if (verbose_ >=1) std::cout << "Escape from y down!" << std::endl;
        return true;
      }      
    }
  }
  return false;  // failed to escape
}

bool Planner::CheckNavGoal(Eigen::Vector2d &sub_goal,
                           const Eigen::Vector2d &nav_goal,
                           const Eigen::Vector2d &pos,
                           const std::vector<Eigen::Vector2d> &vertices) {
  // boundray fix
  float distance = (sub_goal - pos).norm();
  if (distance < 0.1) return false;
  if (abs(sub_goal(0)) > 4.7 || abs(sub_goal(1)) > kHalfMapHeight) {
    if (!this->SimpleRayCast(sub_goal, pos)) return false;
    if (verbose_ >= 1) {
      std::cout << "Wall, sub goal changed to " << sub_goal.transpose() << 
          std::endl;
    }
  }
  // polygon fix
  if (PointInPloy(sub_goal.x(), sub_goal.y(), vertices)) {
    Eigen::Vector2d vec = nav_goal - pos;
    Eigen::Vector2d grad = Eigen::Vector2d(-vec.y(), vec.x()).normalized();
    Eigen::Vector2d left_candidate = sub_goal;
    Eigen::Vector2d right_candidate = sub_goal;
    const double step = 0.1;
    while (true) {
      left_candidate += step * grad;
      if (abs(left_candidate(0)) >= 4.7 || 
          abs(left_candidate(1)) >= kHalfMapHeight) {
        left_candidate = {99.9, 99.9};  // as a large num
        break;
      }
      if (PointInPloy(left_candidate.x(), left_candidate.y(), vertices)) {
        continue;
      }
      // a collision-free pt outside poly is found
      cv::Point pixel = MapCoord2ImgIdx(left_candidate);
      if (cost_map_.at<u_char>(pixel.y, pixel.x) >= 100) break;
    }
    std::cout << "2" << std::endl;
    while (true) {
      right_candidate -= step * grad;
      if (abs(right_candidate(0)) >= 4.7 || 
          abs(right_candidate(1)) >= kHalfMapHeight) {
        right_candidate = {99.9, 99.9};
        break;
      }
      if (PointInPloy(right_candidate.x(), right_candidate.y(), vertices)) {
        continue;
      }
      // a collision-free pt outside poly is found
      cv::Point pixel = MapCoord2ImgIdx(right_candidate);
      if (cost_map_.at<u_char>(pixel.y, pixel.x) >= 100) break;
    }

    if (left_candidate.norm() > 90.0f && right_candidate.norm() > 90.0f) {
      return false;
    }

    double left_distance = (left_candidate - sub_goal).norm();
    double right_distance = (right_candidate - sub_goal).norm();
    if (left_distance < right_distance) {
      sub_goal = left_candidate;
    } else {
      sub_goal = right_candidate;
    }
    if (verbose_ >= 1) {
      std::cout << "Poly, sub goal changed to " << sub_goal.transpose() << 
          std::endl;
    }
  }

  cv::Point pixel = MapCoord2ImgIdx(sub_goal);
  if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) return true;
  // circular fix
  const int kPointNum = 8;
  const double kRadius = 0.5;
  Eigen::Vector2d revised_goal, candidate_goal;
  distance = 100.0f;
  for (int i = 0; i < kPointNum; i++) {
    candidate_goal = sub_goal + kRadius * Eigen::Vector2d(
        std::sin(M_PI * 2 * i / kPointNum),
        std::cos(M_PI * 2 * i / kPointNum));
    /* skip invalid candidate goal */
    if (abs(candidate_goal(0)) >= 4.7 || 
        abs(candidate_goal(1)) >= kHalfMapHeight) {
      continue;
    }

    pixel = MapCoord2ImgIdx(candidate_goal);
    if (cost_map_.at<u_char>(pixel.y, pixel.x) < 100) continue;
    if ((candidate_goal - pos).norm() <= 0.2f) continue;
    if (PointInPloy(candidate_goal(0), candidate_goal(1), vertices)) continue;
    
    double candidate_distance = (candidate_goal - sub_goal).norm();
    if (candidate_distance >= distance) continue;
    /* update revised goal to the nearest */
    distance = candidate_distance;
    revised_goal = candidate_goal;
  }
  if (distance >= 90.0f) {
    return false;
  } else {
    sub_goal = revised_goal;
    if (verbose_ >= 1) {
      std::cout << "Sub goal changed to " << sub_goal.transpose() << std::endl;
    }
    return true;
  }
}

void Planner::UpdateCostMap(const JointState &state) {
  // TODO:
  cost_map_ = map_.clone();

  std::vector<std::vector<cv::Point>> wall_contours;
  // draw walls
  for (const auto &wall : state.walls) {
    std::vector<Eigen::Vector2d> cur_wall;
    cur_wall.push_back({wall.first.x, wall.first.y});
    cur_wall.push_back({wall.second.x, wall.second.y});
    std::vector<cv::Point> cur_contour;
    robot_plann::Vertices2Contour(cur_wall, cur_contour);
    wall_contours.push_back(cur_contour);
  }
  cv::drawContours(cost_map_, wall_contours, -1, cv::Scalar(0), 1);

  std::vector<cv::Point> contour;
  Vertices2Contour(state.rect.vertices, contour);
  std::vector<std::vector<cv::Point>> contours = {contour};
  if (!contour.empty()) {
    cv::drawContours(cost_map_, contours, -1, cv::Scalar(0), 1);
  }

  for (auto &iter: state.obst) {
    cv::circle(cost_map_, MapCoord2ImgIdx({iter.px, iter.py}), 
               (int)round(iter.radius / kMapResol), cv::Scalar(0), 1);
  }

  visual_map_ = cost_map_.clone();
  static cv::Size vis_size(kVisualScale * cost_map_.cols, 
                           kVisualScale * cost_map_.rows);
  
  cv::resize(visual_map_, visual_map_, vis_size);
  cv::cvtColor(visual_map_, visual_map_, cv::COLOR_GRAY2BGR);
  // map inflation
  static int inflation_pixel = (int)round(kInflationRadius / kMapResol) + 2;  // + noise fix 2 pixel
  
  cv::Mat kernal = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(inflation_pixel, inflation_pixel));
  cv::erode(cost_map_, cost_map_, kernal);

  // cv::imshow("debug!", cost_map_);
  // cv::waitKey(5);
}

bool Planner::SimpleRayCast(Eigen::Vector2d &goal, 
                            const Eigen::Vector2d &pos) {
  Eigen::Vector2d ray = goal - pos;
  double distance = ray.norm();
  goal = pos;

  for (double step = 0.0; step < distance; step += 0.1) {
    double t = step / distance;
    Eigen::Vector2d candidate = pos + t * ray;
    // check if the candidate pt is out of bounds
    if (abs(candidate(0)) > 4.5 || abs(candidate(1)) >= kHalfMapHeight) break;
    goal = candidate;
  }

  return ((goal - pos).norm() < 0.1)? false: true;
}

Eigen::Vector2d Planner::PidCalc(const JointState &state) {
  Eigen::Vector3d robot_pose;
  robot_pose << state.robot.px, state.robot.py, state.robot.yaw;
  
  Eigen::Vector2d desired_pos;
  desired_pos << state.robot.gx, state.robot.gy;

  double alpha = 
      atan2(desired_pos(1) - robot_pose(1), desired_pos(0) - robot_pose(0)) - 
      robot_pose(2);
  
  Unwrap(alpha);

  double forward = (alpha <= M_PI / 2 && alpha > -M_PI / 2)? 1.0: -1.0;
  double dist_error = forward * 
      EuclideanNorm(desired_pos(0) - robot_pose(0), 
                    desired_pos(1) - robot_pose(1));

  if (abs(dist_error) < 0.05) alpha = 0;

  Eigen::Vector2d acc_ctrl;
  Eigen::Vector2d desired_vel;
  desired_vel << 0.8 * dist_error, 2.0 * alpha;

  acc_ctrl(0) = (desired_vel(0) - state.robot.v) / (kDT * 0.5);
  acc_ctrl(1) = (desired_vel(1) - state.robot.yaw_rate) / (kDT * 0.5);

  return acc_ctrl;
}

cv::Mat Planner::CreateMap() {
  cv::Mat map(400, 240, CV_8UC1, cv::Scalar(255));  // TODO: change to 10 x 20

  std::cout << "create map" << std::endl;
  return std::move(map);
};

robot_plann::MPCOutputForPython Planner::RunSlover(
  const robot_plann::MPCInputForPython& input) {
  
  robot_plann::MPCOutputForPython ans{};
  ans.astar_path.clear();
  ans.control_vars.clear();
  ans.success = false;
  if (input.valid == false) {
    return ans;
  }
  robot_plann::JointState ob_state;

  PybindInputDataChange(input, ob_state);
  Eigen::Vector2d sub_goal = {input.sub_goal.x, input.sub_goal.y};

  auto start_stamp = std::chrono::high_resolution_clock::now();
  auto mpc_return = this->PlannExec(ob_state, sub_goal);
  auto end_stamp = std::chrono::high_resolution_clock::now();
  double time_cost =
      std::chrono::duration<double, std::milli>(end_stamp - start_stamp)
          .count();

  std::vector<robot_plann::Point> astar_path = this->GetAStarPath();
  for (int i = 0; i < astar_path.size(); ++i) {
    robot_plann::Point pt;
    pt.x = astar_path.at(i).x;
    pt.y = astar_path.at(i).y;
    pt.v = 0.0;
    ans.astar_path.push_back(pt);

  }

  robot_plann::ControlVar cur_control_var{};
  if (!mpc_return.success) {
    ans.al = 0.0;
    ans.ar = 0.0;
    ans.revised_goal.x = input.sub_goal.x;
    ans.revised_goal.y = input.sub_goal.y;
    std::cout << "Ocp plann failed!" << std::endl;
    cur_control_var.al = ans.al;
    cur_control_var.ar = ans.ar;
    for (int i = 0; i < kNP; ++i) {
      ans.control_vars.push_back(cur_control_var);
    }

  } else {
    ans.al = mpc_return.stages[0].uk.acc - mpc_return.stages[0].uk.dr * 0.3;
    ans.ar = mpc_return.stages[0].uk.acc + mpc_return.stages[0].uk.dr * 0.3;
    ans.revised_goal.x = input.sub_goal.x;
    ans.revised_goal.y = input.sub_goal.y;
   
    for (int i = 0; i < kNP; ++i) {
      cur_control_var.al =
          mpc_return.stages.at(i).uk.acc - mpc_return.stages.at(i).uk.dr * 0.3;
      cur_control_var.ar =
          mpc_return.stages.at(i).uk.acc + mpc_return.stages.at(i).uk.dr * 0.3;

      ans.control_vars.push_back(cur_control_var);
    }
    ans.success = true;
    std::cout << "Time cost: " << time_cost << std::endl;

  }

  return std::move(ans);
}


void Planner::PybindInputDataChange(const robot_plann::MPCInputForPython& input,
                               robot_plann::JointState &ob_state) {
  
  const auto &robot_state = input.ob.robot;
  ob_state.robot.px = robot_state.px;
  ob_state.robot.py = robot_state.py;
  ob_state.robot.yaw = robot_state.yaw;
  ob_state.robot.v = robot_state.v;
  ob_state.robot.yaw_rate = robot_state.yaw_rate;

  ob_state.robot.v_pref = robot_state.v_pref;
  ob_state.robot.radius = robot_state.radius;
  ob_state.robot.gx = robot_state.gx;
  ob_state.robot.gy = robot_state.gy;

  for (const auto &hum_iter : input.ob.hum) {
    robot_plann::HumanState hum_state;
    hum_state.px = hum_iter.px;
    hum_state.py = hum_iter.py;
    hum_state.vx = hum_iter.vx;
    hum_state.vy = hum_iter.vy;
    hum_state.radius = hum_iter.radius;

    ob_state.hum.push_back(hum_state);
  }

  for (const auto &obst_iter : input.ob.obst) {
    robot_plann::ObstacleState obst_state;
    obst_state.px = obst_iter.px;
    obst_state.py = obst_iter.py;
    obst_state.radius = obst_iter.radius;
    ob_state.obst.push_back(obst_state);
  }

  ob_state.rect.vertices.clear();
  // for (const auto &poly_iter : input.ob.rect) {
  for (const auto &vertex : input.ob.rect.vertices) {
    Eigen::Vector2d pt(vertex.x, vertex.y);
    ob_state.rect.vertices.push_back(pt);
  }

  ob_state.walls.clear();
  for (const auto &input_wall : input.ob.walls) {
    Wall wall = ClipWall(input_wall.sx, input_wall.sy, input_wall.ex, input_wall.ey, -5.5, 5.5);

    ob_state.walls.push_back(wall);
  }

}

Wall Planner::ClipWall(double x1, double y1, double x2, double y2, double x_min, double x_max) {
    Wall ans;
    bool swap_flag = false;
    if (x1 > x2) {
      std::swap(x1, x2);
      std::swap(y1, y2);
      swap_flag = true;
    }

    if (x2 < x_min || x1 > x_max) {
        std::cerr << "===error wall===" << std::endl;
        return ans;
    }

    double slope = 0;
    if (x1 != x2) {
        slope = (y2 - y1) / (x2 - x1);
    } else {
      if (x1 < x_min || x1 > x_max) {
        std::cerr << "===error wall===" << std::endl;
        return ans;
      } else {
        ans.first.x = swap_flag ? x2 : x1;
        ans.first.y = swap_flag ? y2 : y1;
        ans.second.x = swap_flag ? x1 : x2;
        ans.second.y = swap_flag ? y1 : y2;
        return ans;
      }
    }

    if (x1 < x_min) {
        y1 = y1 + slope * (x_min - x1);
        x1 = x_min;
    }

    if (x2 > x_max) {
        y2 = y1 + slope * (x_max - x1);
        x2 = x_max;
    }

    ans.first.x = swap_flag ? x2 : x1;
    ans.first.y = swap_flag ? y2 : y1;
    ans.second.x = swap_flag ? x1 : x2;
    ans.second.y = swap_flag ? y1 : y2;
    return ans;
}

}  // namespace robot_plann