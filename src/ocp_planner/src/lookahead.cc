/**
  ******************************************************************************
  * @file    lookahead.cc
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2021/12/27
  * @brief   general motion planning for MobiRo @ tib_k331
  ******************************************************************************
  * @attention modified from ICRA2021 RMUA code by HITSZ 
  *
  ******************************************************************************
  */

#include "lookahead.h"

namespace robot_plann {
//
std::vector<struct Point> LookAhead::UpdateVelocity(
    std::vector<SmoothPath>& path, double current_vel) {
  std::vector<struct Point> final_path;
  path.front().start_vel_ = current_vel;
  path.back().end_vel_ = 0;  // terminal vel set to zero

  for (auto iter = path.begin() + 1; iter != path.end(); iter += 2) {
    if(!iter->is_arc_) continue;  // skip line segment
    // double candidate_vel = iter->radius_ * omega_max_;
    double candidate_vel = iter->radius_ * 3.0;  // needs properly adjust

    double prev_start_vel = (iter - 1)->start_vel_;
    double prev_len = (iter - 1)->length_;
    /* vel bias between segment is too large, candidate needs adjusted */
    if (AccelDistance(candidate_vel, prev_start_vel, a_max_) > prev_len) {
      if (candidate_vel > prev_start_vel) {
        candidate_vel = 
            std::sqrt(square(prev_start_vel) + 2 * a_max_ * prev_len);  // accel
      } else {
        candidate_vel = 
            std::sqrt(square(prev_start_vel) - 2 * a_max_ * prev_len);  // decel
      }
    }
    UpperBound(candidate_vel, v_max_);
    LowerBound(candidate_vel, 0.0);

    (iter - 1)->end_vel_ = candidate_vel;
    (iter + 1)->start_vel_ = candidate_vel;
    iter->start_vel_ = candidate_vel;  // the same 
    iter->end_vel_ = candidate_vel;
  }

  /* unable to decel to zero in the last segment */
  if (path.size() >= 3 && 
      square(path.back().start_vel_) > 2 * a_max_ * path.back().length_) {
    for (auto iter = path.end() - 2; iter != path.begin(); iter -= 2) {
      double next_start_vel = (iter + 1)->start_vel_;
      double next_end_vel = (iter + 1)->end_vel_;
      double next_len = (iter + 1)->length_;

      if (square(next_start_vel) - square(next_end_vel) <= \
          2 * a_max_ * next_len) break;
      /* vel trace back */
      double candidate_vel = std::sqrt(
          square(next_end_vel) + 2 * a_max_ * next_len);

      UpperBound(candidate_vel, v_max_);
      (iter + 1)->start_vel_ = candidate_vel;
      iter->start_vel_ = candidate_vel;
      iter->end_vel_ = candidate_vel;
      (iter - 1)->end_vel_ = candidate_vel;
    }
  }

  double point_step_len = 0.2;
  for (auto iter = path.begin(); iter != path.end(); iter++) {
    double segment_len_sum = 0;
    while (segment_len_sum < iter->length_) {
      LowerBound(iter->length_, 0.1);  // avoid singularity
      segment_len_sum += point_step_len;
      UpperBound(segment_len_sum, iter->length_);

      struct Point candidate;
      TrapezoidalParams params;
      params.v_start = iter->start_vel_;
      params.v_end = iter->end_vel_;
      params.v_max = v_max_;
      params.accel = a_max_;
      params.d = iter->length_ - segment_len_sum;
      params.l = iter->length_;

      if (iter->is_arc_ == false) { // line final path generate
        /* in case segment_len_sum is equal to iter->length_ 
          candidate coord will become segment end point */
        candidate.x = (iter->end_x_ - iter->start_x_) * \
                      segment_len_sum / iter->length_ + iter->start_x_;
        candidate.y = (iter->end_y_ - iter->start_y_) * \
                      segment_len_sum / iter->length_ + iter->start_y_;
        candidate.v = this->TrapezoidalPlan(params);
      } else {  // arc final path generate
        LowerBound(iter->radius_, 0.1); 
        double delta_angle = iter->end_angle_ - iter->start_angle_;
        if (std::fabs(delta_angle) < 0.1) delta_angle = 0.1;  // avoid singularity
        Unwrap(delta_angle);

        double arc_step = 
            segment_len_sum / (iter->radius_ * myabs(delta_angle));                
        double angle_step = (iter->end_angle_ - iter->start_angle_) * \
                            arc_step + iter->start_angle_;                     
        candidate.x = iter->center_x_ + iter->radius_ * cos(angle_step);
        candidate.y = iter->center_y_ + iter->radius_ * sin(angle_step);
        candidate.v = this->TrapezoidalPlan(params);
      }
      final_path.push_back(candidate);
    }
  }
  if (final_path.size() != 0) final_path.back().v = 0;
  // simple coord trans
  for (auto &pt: final_path) {
    pt.x -= kHalfMapWidth;
    pt.y -= kHalfMapHeight;
  }
  return final_path;
}
/* (params.l - params.d) represent segment_len_sum */
double LookAhead::TrapezoidalPlan(TrapezoidalParams &params) {
  if (params.l < 0.01) return 0;
  if (params.l - params.d < 0) params.d = params.l - 0.01;
  if (params.d < 0) params.d = 0;

  double acc_dis = AccelDistance(params.v_max, params.v_start, params.accel);
  double dec_dis = AccelDistance(params.v_max, params.v_end, params.accel);

  if (acc_dis + dec_dis < params.l) {  // can reach v_max
    if (params.l - params.d < acc_dis) {  // accel phase
      return std::sqrt(  
          square(params.v_start) + 2 * params.accel * (params.l - params.d));
      
    } else if (params.l - params.d < params.l - dec_dis) {  // max vel phase
      return params.v_max;
    } else {  // decel phase  
      return std::sqrt(square(params.v_end) + 2 * params.accel * params.d);
    }    
  } else {  // v_max can not be reached
    double dis = AccelDistance(params.v_start, params.v_end, params.accel);
    if (dis < params.l) {  // still have both accel and decel phase
      double v_max_local = std::sqrt((square(params.v_start) + \
          square(params.v_end) + 2 * params.accel * params.l) / 2);
      acc_dis = AccelDistance(params.v_start, v_max_local, params.accel);
      // accel phase
      if (params.l - params.d < acc_dis) {
        return std::sqrt(
            square(params.v_start) + 2 * params.accel * (params.l - params.d));
      } else {
        return std::sqrt(square(params.v_end) + 2 * params.accel * params.d);
      }
    } else {  // decel or accel only 
      if (params.v_start > params.v_end) {
        return std::sqrt(square(params.v_end) + 2 * params.accel * params.d);
      } else {
        return std::sqrt(
            square(params.v_start) + 2 * params.accel * (params.l - params.d));
      }
    }
  }
}
//
} // namesapce tib_k331_planning
