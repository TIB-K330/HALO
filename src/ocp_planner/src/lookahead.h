/**
  ******************************************************************************
  * @file    lookahead.h
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2021/12/27
  * @brief   general motion planning for MobiRo @ tib_k331
  ******************************************************************************
  * @attention modified from ICRA2021 RMUA code by HITSZ 
  *
  ******************************************************************************
  */
#ifndef LOOKAHEAD_H
#define LOOKAHEAD_H

#include "smooth.h"

namespace robot_plann {
//
struct TrapezoidalParams {
  double v_start;
  double v_end;
  double v_max;
  double accel;
  double d;  // unknown, maybe delay
  double l;
};

class LookAhead {
 public:
  LookAhead(): v_max_(2.0), a_max_(4.0), omega_max_(3.14) {}
  ~LookAhead() = default;

  void SetParams(double v_max, double a_max, double omega_max) {
    v_max_ = v_max;
    a_max_ = a_max;
    omega_max_ = omega_max; 
  }

  auto UpdateVelocity(std::vector<SmoothPath>& path, double current_vel) \
      -> std::vector<struct Point>;

 private:
  double TrapezoidalPlan(TrapezoidalParams &params);

  double v_max_;
  double a_max_;
  double omega_max_;
};

}  // namespace robot_plann

#endif  // LOOKAHEAD_H