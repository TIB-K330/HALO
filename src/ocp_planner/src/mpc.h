/**
  ******************************************************************************
  * @file    mpc.h
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2023/05/05
  * @brief   ocp planner as rl backend for MobiRo @ tib_k331
  ******************************************************************************
  * @attention
  *
  ******************************************************************************
  */
#ifndef MPC_H
#define MPC_H

#include <casadi/casadi.hpp>
#include "types.h"
#include "utils.h"

namespace robot_plann {

inline bool CompareObstacleDistance(const robot_plann::DynaObstacle &obst1, 
                                    const robot_plann::DynaObstacle &obst2) { 
  return obst1.distance < obst2.distance;
}

class Mpc {
 public:
  explicit Mpc(int verbose = 0): verbose_(verbose) {
    // init default mpc params
    params_.reset(new MpcParams);
    params_->dt = kDT;
    params_->np = kNP;
    params_->max_linear_vel = 1.0;
    params_->max_linear_acc = 1.0;
    params_->max_angular_vel = 3.0;
    params_->max_angular_acc = 3.0;
    params_->local_obst_num = 6;
  }

  void SetParams(MpcParams::Ptr params) {
    params_ = params;
  }

  MpcReturn RunMpc(const JointState &state, 
                   std::vector<robot_plann::Point> &path);

 private:
  void ObstHerp(const std::vector<int> &vOb,
                const std::vector<std::vector<Eigen::Vector2d>> &lOb,
                Eigen::MatrixXd &A, Eigen::MatrixXd &B);

  void RotateState(const JointState &state, 
                   std::vector<robot_plann::Point> &path);

  int UpdateReferencePath(const std::vector<robot_plann::Point> &path,
                          std::vector<robot_plann::Point> &local_ref,
                          const Eigen::Vector2d &robot_pos);

  void OrderObstacles(robot_plann::ObstacleArray &obst_arr_msg);

  void PredictObstacleStates(const JointState &state, 
                             robot_plann::ObstacleArray &obst_arr_msg);

  void NewInitGuess(const RobotState &robot_state,
                    const std::vector<robot_plann::Point> local_ref, 
                    MpcStages &init_guess);

  casadi::Function RobotModelDifferential();
  
  casadi::Function DynamicFE(casadi::Function &model, double dt);

  casadi::Function DynamicRK2(casadi::Function &model, double dt);
  
  casadi::Function DynamicRK4(casadi::Function &model, double dt);
  
  int SolveMpc(const Eigen::MatrixXd &A, const Eigen::MatrixXd &b,
               const robot_plann::ObstacleArray &obst_arr_msg,
               const std::vector<robot_plann::Wall> &walls,
               MpcStages &init_guess, const OptVarIndex &var_idx);

  MpcParams::Ptr params_;
  int verbose_;
};

}  // namespace robot_plann

#endif  // MPC_H
