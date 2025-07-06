/**
  ******************************************************************************
  * @file    mpc.cc
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2023/05/05
  * @brief   ocp planner as rl backend for MobiRo @ tib_k331
  ******************************************************************************
  * @attention
  *
  ******************************************************************************
  */
#include "mpc.h"

namespace robot_plann {

MpcReturn Mpc::RunMpc(const JointState &state, 
                      std::vector<robot_plann::Point> &path) {
  Eigen::MatrixXd AOb, bOb;
  if (state.rect.vertices.size()) {
    std::vector<int> vOb = {(int)state.rect.vertices.size() + 1,};
    std::vector<std::vector<Eigen::Vector2d>> lOb;
    lOb.emplace_back(state.rect.vertices);
    for (auto &iter: lOb) {
      iter.emplace_back(iter.front());
    }
    
    this->ObstHerp(vOb, lOb, AOb, bOb);
  } else {
    AOb.resize(0, 0);
    bOb.resize(0, 0);
  }

  OptVarIndex var_idx = {5, (int)AOb.rows(), 0, 2};
  Eigen::Vector2d robot_pos; 
  robot_pos << state.robot.px, state.robot.py;
  std::vector<robot_plann::Point> local_ref;
  int path_idx = this->UpdateReferencePath(path, local_ref, robot_pos);
  MpcStages opt_traj;
  if (path_idx < 2) return {opt_traj, false};

  this->NewInitGuess(state.robot, local_ref, opt_traj);

  robot_plann::ObstacleArray obst_array;
  PredictObstacleStates(state, obst_array);
  
  if (this->SolveMpc(AOb, bOb, obst_array, state.walls, opt_traj, var_idx) == 0) {
    return {opt_traj, true};
  } else {
    return {opt_traj, false};
  }
}

void Mpc::RotateState(const JointState &state, 
                      std::vector<robot_plann::Point> &path) {
  const float rot_ang = state.robot.yaw;                      
  Eigen::Matrix2d trans_matrix;
  trans_matrix << cos(rot_ang), -sin(rot_ang),
                  sin(rot_ang),  cos(rot_ang);
  /* TODO: Trans. to the robot-cetric coord. to address potential 
      singularity caused by the orientation angles */
}

int Mpc::UpdateReferencePath(const std::vector<robot_plann::Point> &path,
                             std::vector<robot_plann::Point> &local_ref,
                             const Eigen::Vector2d &robot_pos) {
  if (path.size() < 2) {
    std::cout << "Invalid global ref. path!!!" << std::endl;
    return -1;
  }

  if (path.size() > params_->np) {
    local_ref = std::vector<robot_plann::Point>(path.begin(), 
                                                path.begin() + params_->np);
  } else {
    local_ref = path;
  }

  return local_ref.size();
}

void Mpc::OrderObstacles(robot_plann::ObstacleArray &obst_arr_msg) {
  if (!obst_arr_msg.dyna_obstacles.size()) return;
  auto &obst_array = obst_arr_msg.dyna_obstacles;
  std::sort(obst_array.begin(), obst_array.end(), CompareObstacleDistance);
}

void Mpc::PredictObstacleStates(const JointState &state,
                                robot_plann::ObstacleArray &obst_arr_msg) {
  auto &obst_array = obst_arr_msg.dyna_obstacles;
  obst_array.clear();

  for (int i = 0; i < state.hum.size(); i++) {
    robot_plann::DynaObstacle obst;

    obst.distance = EuclideanNorm(state.robot.px - state.hum[i].px, 
                                  state.robot.py - state.hum[i].py);

    if (obst.distance > 4.0) continue;
    obst.radius = state.hum.at(i).radius;
    // obst.trajectory.poses.resize(params_->np);
    obst.pred_trajectorys.resize(1);
    obst.pred_trajectorys.at(0).resize(params_->np);
    for (int k = 0; k < params_->np; k++) {
      auto &pos = obst.pred_trajectorys.at(0).at(k);

      pos.x = state.hum[i].px + k * params_->dt * state.hum[i].vx;
      pos.y = state.hum[i].py + k * params_->dt * state.hum[i].vy;
      pos.v = 1.0;

    }
    obst_array.push_back(obst);
  }

  for (int i = 0; i < state.obst.size(); i++) {
    robot_plann::DynaObstacle obst;
    obst.distance = EuclideanNorm(state.robot.px - state.obst[i].px, 
                                  state.robot.py - state.obst[i].py);
    if (obst.distance > 4.0) continue;

    obst.radius = state.obst.at(i).radius;
    // obst.trajectory.poses.resize(params_->np);
    obst.pred_trajectorys.resize(1);
    obst.pred_trajectorys.at(0).resize(params_->np);

    for (int k = 0; k < params_->np; k++) {
      // auto &pos = obst.trajectory.poses[k].pose.position;
      auto &pos = obst.pred_trajectorys.at(0).at(k);
      pos.x = state.obst[i].px;
      pos.y = state.obst[i].py;
      pos.v = 1.0;
    }
 
    obst_array.push_back(obst);
  }

  this->OrderObstacles(obst_arr_msg);

  for (int i = params_->local_obst_num; i < obst_array.size(); i++) {
    obst_array.pop_back();
  }
}

void Mpc::NewInitGuess(const RobotState &robot_state,
                       const std::vector<robot_plann::Point> local_ref,
                       MpcStages &init_guess) {
  init_guess.at(0).xk.setZero(0, 0);
  init_guess.at(0).uk.setZero();

  init_guess.at(0).xk.X = robot_state.px;
  init_guess.at(0).xk.Y = robot_state.py;
  init_guess.at(0).xk.phi = robot_state.yaw;
  init_guess.at(0).xk.vx = robot_state.v;
  init_guess.at(0).xk.r = robot_state.yaw_rate;

  double phi = M_PI / 2;
  for (int i = 1; i < local_ref.size(); i++) {
    init_guess.at(i).xk.setZero(0, 0);
    init_guess.at(i).uk.setZero();

    init_guess.at(i).xk.X  = local_ref.at(i).x;
    init_guess.at(i).xk.Y  = local_ref.at(i).y;
    init_guess.at(i).xk.vx = local_ref.at(i).v;

    if (i != local_ref.size() - 1) {
      phi = atan2(local_ref[i + 1].y - local_ref[i].y,
                  local_ref[i + 1].x - local_ref[i].x);
    } else {
      phi = atan2(local_ref[i].y - local_ref[i - 1].y,
                  local_ref[i].x - local_ref[i - 1].x);
    } 
    init_guess.at(i).xk.phi = phi;
  }
  // fill-in with the last ref. pt
  for (int i = local_ref.size(); i < params_->np; i++) {
    init_guess.at(i).xk.setZero(0, 0);
    init_guess.at(i).uk.setZero();

    init_guess.at(i).xk.X  = local_ref.back().x;
    init_guess.at(i).xk.Y  = local_ref.back().y;
    init_guess.at(i).xk.vx = local_ref.back().v;
    init_guess.at(i).xk.phi = phi;
  }

}

casadi::Function Mpc::RobotModelDifferential() {
  auto x = casadi::MX::sym("X", 5, 1);
  auto u = casadi::MX::sym("U", 3, 1);
  
  casadi::MX x_dot = casadi::MX::vertcat({x(3) * cos(x(2)),
                                          x(3) * sin(x(2)),
                                          x(4),
                                          u(0),
                                          u(1)});

  return casadi::Function("continuous_dynamic", {x, u}, {x_dot});
}

casadi::Function Mpc::DynamicFE(casadi::Function &model, double dt) {
  auto x = casadi::MX::sym("X", 5, 1);
  auto u = casadi::MX::sym("U", 3, 1);

  casadi::MX x_next = x + model(casadi::MXVector{x, u})[0] * dt;
  
  return casadi::Function("forward_euler", {x, u}, {x_next});
}

casadi::Function Mpc::DynamicRK2(casadi::Function &model, double dt) {
  auto x = casadi::MX::sym("X", 5, 1);
  auto u = casadi::MX::sym("U", 3, 1);

  casadi::MXVector k1 = model(casadi::MXVector{x, u});
  casadi::MXVector k2 = model(casadi::MXVector{x + dt * k1[0], u});
  
  casadi::MX x_next = x + (dt / 2.0) * (k1[0] + k2[0]);
  
  return casadi::Function("dynamic_rk2", {x, u}, {x_next});
}

casadi::Function Mpc::DynamicRK4(casadi::Function &model, double dt) {
  auto x = casadi::MX::sym("X", 5, 1);
  auto u = casadi::MX::sym("U", 3, 1);

  casadi::MXVector k1 = model(casadi::MXVector{x, u});
  casadi::MXVector k2 = model(casadi::MXVector{x + dt * k1[0] * 0.5, u});
  casadi::MXVector k3 = model(casadi::MXVector{x + dt * k2[0] * 0.5, u});
  casadi::MXVector k4 = model(casadi::MXVector{x + dt * k3[0], u});
  auto x_next = x + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
  
  return casadi::Function("dynamic_rk4", {x, u}, {x_next});
}

int Mpc::SolveMpc(const Eigen::MatrixXd &A, const Eigen::MatrixXd &b,
                  const robot_plann::ObstacleArray &obst_arr_msg,
                  const std::vector<robot_plann::Wall> &walls,
                  MpcStages &init_guess, const OptVarIndex &var_idx) {
  /* opt var idx
    0 state.X
    1 state.Y
    2 state.phi
    3 state.vx
    4 state.r
    *** */
  casadi::Opti opti = casadi::Opti("nlp");  // Optimization problem
  const int &np = init_guess.size();
  assert(np == params_->np);
  // define variables
  const int ext_state_num = var_idx.state_var_num + var_idx.obst_dual_num;
  auto X = opti.variable(ext_state_num, np);
  auto U = opti.variable(var_idx.input_var_num + 1, np);  // +1 represent slack variable
  // cost function 
  auto f = opti.f();
  // min slack var
  for (int k = 0; k < np; k++) f += 99999 * pow(U(2, k), 2);
  // min pose error
  for (int k = 0; k < np - 1; k++) {
    f += 5.0 * pow(X(0, k) - init_guess.at(k).xk.X, 2);  // x_err
    f += 5.0 * pow(X(1, k) - init_guess.at(k).xk.Y, 2);  // y_err
    // f += 2.0 * pow(X(3, k) - init_guess.at(k).xk.vx, 2);  // v_err
    f += 1.0 * pow(X(4, k), 2);
  }
  // terminal cost 
  f += 100 * pow(X(0, np - 1) - init_guess.back().xk.X, 2);
  f += 100 * pow(X(1, np - 1) - init_guess.back().xk.Y, 2);
  // f += 200 * pow(X(3, np - 1), 2);
  // input regularization
  for (int k = 0; k < np - 1; k++) {
    f += 2 * pow(U(0, k), 2);
    f += 0.5 * pow(U(1, k), 2);
  }
  
  for (int k = 0; k < np - 2; k++) {
    f += 0.05 * pow(U(0, k + 1) - U(0, k), 2);
    f += 0.05 * pow(U(1, k + 1) - U(1, k), 2);
  }

  opti.minimize(f);

  // robot dynamic constraints aka robot model
  const double &dt = params_->dt;
  auto continuous_model = RobotModelDifferential();
  auto discrete_model = this->DynamicRK2(continuous_model, dt);  // a trade-off

  for (int k = 0; k < np - 1; k++) {
    casadi::MX X_k = X(casadi::Slice(0, var_idx.state_var_num), k);
    casadi::MX U_k = U(casadi::Slice(), k);
    casadi::MX X_next = discrete_model(casadi::MXVector{X_k, U_k})[0];
    opti.subject_to(
        X(casadi::Slice(0, var_idx.state_var_num), k + 1) == X_next);
  }

  // init constraints
  opti.subject_to(X(0, 0) == init_guess.front().xk.X);
  opti.subject_to(X(1, 0) == init_guess.front().xk.Y);
  opti.subject_to(X(2, 0) == init_guess.front().xk.phi);
  opti.subject_to(X(3, 0) == init_guess.front().xk.vx);
  opti.subject_to(X(4, 0) == init_guess.front().xk.r);

  // input constraints
  for (int k = 0; k < np; k++) {
    // left walls right_up to left_down
    // right walls left_down to right_up
    if (!walls.empty()) {
      if (walls.size() != 2) {
        std::cerr << "wall is not set true!" << std::endl;
      }
      for (int l = 0; l < 2; ++l) {
        const Wall &wall = walls.at(l);
        const double sx = wall.first.x;
        const double sy = wall.first.y;
        const double dx = wall.second.x - wall.first.x;
        const double dy = wall.second.y - wall.first.y;
        opti.subject_to(dx * (X(1, k) - sy) - dy * (X(0, k) - sx) >= 0.0);
        // opti.subject_to(dx * (X(1, k) - sy) - dy * (X(0, k) - sx) >= 0.11);
      }
    }
    // opti.subject_to(-4.7 <= X(0, k) <= 4.7);
    opti.subject_to(-1.0 <= X(3, k) - X(4, k) * 0.3 <= 1.0);
    opti.subject_to(-1.0 <= X(3, k) + X(4, k) * 0.3 <= 1.0);
    opti.subject_to(-1.0 <= U(0, k) - U(1, k) * 0.3 <= 1.0);
    opti.subject_to(-1.0 <= U(0, k) + U(1, k) * 0.3 <= 1.0);
    opti.subject_to(U(2, k) >= 0);  // s_k >= 0
  }

  /* dynamic obstacle avoidance */
  const auto &obst_array = obst_arr_msg.dyna_obstacles;
  auto obst_params = opti.parameter(3 * obst_array.size(), np);
  // auto obst_params = opti.parameter(3 * obst_array.size(), (np - 3));

  // TODO: Currently, only the path with the highest probability is used.
  for (int i = 0; i < (int)obst_array.size(); i++) {
    const double safe_dist = obst_array.at(i).radius + kInflationRadius + 0.1;
    // const double safe_dist = obst_array.at(i).radius + kInflationRadius + 0.21;

    // const auto &pred_obst_traj = obst_array.at(i).trajectory;
    const auto &pred_obst_traj = obst_array.at(i).pred_trajectorys.at(0);
    // for (int k = 0; k < (np - 3); k++) {
    for (int k = 0; k < np; k++) {

      // const auto &pos = pred_obst_traj.poses[k].pose.position;
      const auto &pos = pred_obst_traj.at(k);

      opti.set_value(obst_params(3 * i, k), pos.x);
      opti.set_value(obst_params(3 * i + 1, k), pos.y);
      opti.set_value(obst_params(3 * i + 2, k), pow(safe_dist, 2));

      auto square_dis = pow(X(0, k) - obst_params(3 * i, k), 2) + 
                        pow(X(1, k) - obst_params(3 * i + 1, k), 2);
      
      opti.subject_to(square_dis + U(2, k) >= obst_params(3 * i + 2, k));
    }
  }
  /* *** */
  
  /* polygon obstacle avoidance */
  if (A.rows() || b.rows()) {
    // define casadi parameters
    assert(A.rows() == b.rows());
    auto AOb = opti.parameter(A.rows(), 2);
    auto bOb = opti.parameter(b.rows(), 1);

    for (int i = 0; i < (int)A.rows(); i++) {
      opti.set_value(AOb(i, 0), A(i, 0));
      opti.set_value(AOb(i, 1), A(i, 1));
      opti.set_value(bOb(i, 0), b(i, 0));
    }
    // minimum-penetration constraints
    for (int k = 0; k < np; k++) {
      auto pk = X(casadi::Slice(0, 2), k);  // p_k
      auto lk = X(casadi::Slice(var_idx.state_var_num, ext_state_num), k); // lambda_k
      auto norm_vec = mtimes(AOb.T(), lk);
      opti.subject_to((pow(norm_vec(0), 2) + pow(norm_vec(1), 2)) == 1.0);  // dual_norm(A'*lambda_k) == 1
      opti.subject_to(
        mtimes((mtimes(AOb, pk) - bOb).T(), lk) + U(2, k) >= 
        // kInflationRadius + 0.11);  // (A*p_k - b)'*lambda_k > -s_k
        kInflationRadius + 0.1);  // (A*p_k - b)'*lambda_k > -s_k
      
      for (int i = 0; i < var_idx.obst_dual_num; i++) {
        opti.subject_to(lk(i) >= 0.0);  // lambda_k >= 0
      }
    }
  }
  /* *** */
  // initial guess and bounds for the optimization variables
  for (int k = 0; k < init_guess.size(); k++) {
    opti.set_initial(X(0, k), init_guess.at(k).xk.X);
    opti.set_initial(X(1, k), init_guess.at(k).xk.Y);
    opti.set_initial(X(2, k), init_guess.at(k).xk.phi);
    opti.set_initial(X(3, k), init_guess.at(k).xk.vx);
  }

  // create nlp solver and buffers
  casadi::Dict ipopt_opts;
  ipopt_opts["print_level"] = 0;
  ipopt_opts["linear_solver"] = "ma57";
  // ipopt_opts["hessian_approximation"] = "limited-memory";
  ipopt_opts["max_iter"] = 300;
  ipopt_opts["tol"] = 5e-4;
  ipopt_opts["warm_start_init_point"] = "yes";
  ipopt_opts["max_cpu_time"] = 0.1;


  casadi::Dict nlp_opts;

  opti.solver("ipopt", nlp_opts, ipopt_opts);

  try {
    auto start_stamp = std::chrono::high_resolution_clock::now();
    auto sol = opti.solve_limited();
    auto end_stamp = std::chrono::high_resolution_clock::now();
    double time_cost = std::chrono::duration<double, std::milli>(
        end_stamp - start_stamp).count();
    
    if (verbose_ >= 1) {
      std::cout << "Mpc time cost: " << time_cost << std::endl;
    }

    if (sol.stats()["success"]) {
      for (int k = 0; k < init_guess.size(); k++) {

        init_guess.at(k).xk.X   = (double)sol.value(X(0, k));
        init_guess.at(k).xk.Y   = (double)sol.value(X(1, k));
        init_guess.at(k).xk.phi = (double)sol.value(X(2, k));
        init_guess.at(k).xk.vx  = (double)sol.value(X(3, k));
        init_guess.at(k).xk.r   = (double)sol.value(X(4, k));

        init_guess.at(k).uk.acc = (double)sol.value(U(0, k));
        init_guess.at(k).uk.dr  = (double)sol.value(U(1, k));
      }
      return 0;
    } 
    return 1;
  } catch (const casadi::CasadiException &e) {
    std::cerr << "CasADi exception: " << e.what() << std::endl;
    return -1;
  }
}

// !!! vetices given in CLOCK-WISE direction !!!
void Mpc::ObstHerp(const std::vector<int> &vOb,
                   const std::vector<std::vector<Eigen::Vector2d>> &lOb,
                   Eigen::MatrixXd &A, Eigen::MatrixXd &B) {
  if (vOb.size() != lOb.size()) return;
  
  int vertex_num = 0;
  for (auto iter = vOb.begin(); iter != vOb.end(); iter++) {
    vertex_num += *iter;
  }

  // edge_num = vertex_num - obst_num;
  A.resize(vertex_num - vOb.size(), 2);
  B.resize(vertex_num - vOb.size(), 1);

  int edge_counter = 0;  // counter for block idx of matrix A and B
  Eigen::MatrixXd A_i, B_i;

  for (int i = 0; i < (int)vOb.size(); i++) {
    A_i.resize(vOb.at(i) - 1, 2);
    B_i.resize(vOb.at(i) - 1, 1);
    Eigen::Vector2d A_temp; double B_temp = 0.0;
    
    for (int j = 0; j < vOb[i] - 1; j++) {
      Eigen::Vector2d vertex_1 = lOb[i][j];
      Eigen::Vector2d vertex_2 = lOb[i][j + 1];
      /* find hyperplane passing through vertex_1 and vertex_2 */
      if (vertex_1(0) == vertex_2(0)) {  
        // perpendicular hyperplane, not captured by general formula
        if(vertex_2(1) < vertex_1(1)) {   // line goes "down"
          A_temp = Eigen::Vector2d(1, 0);
          B_temp = vertex_1(0);
        } else {  // line goes "up"
          A_temp = Eigen::Vector2d(-1, 0);
          B_temp = -vertex_1(0);
        }
      } else if (vertex_1(1) == vertex_2(1)) {
        // horizontal hyperplane, captured by general formula but included for numerical stability
        if(vertex_1(0) < vertex_2(0)) {
          A_temp = Eigen::Vector2d(0, 1);
          B_temp = vertex_1(1);
        } else {
          A_temp = Eigen::Vector2d(0, -1);
          B_temp = -vertex_1(1);
        }
      } else {  
        // general formula for non-horizontal and non-vertical hyperplanes
        Eigen::Matrix2d mat_temp;
        mat_temp << vertex_1(0), 1, 
                    vertex_2(0), 1;
        
        Eigen::Vector2d vec_temp(vertex_1(1), vertex_2(1));
        Eigen::Vector2d ans = mat_temp.colPivHouseholderQr().solve(vec_temp);
        if(vertex_1(0) < vertex_2(0)) {
          A_temp = Eigen::Vector2d(-ans(0), 1);
          B_temp = ans(1);
        } else {
          A_temp = Eigen::Vector2d(ans(0), -1);
          B_temp = -ans(1);
        }
      }
      A_i.row(j) = A_temp.transpose();
      B_i(j) = B_temp;
    }

    A.block(edge_counter, 0, vOb.at(i) - 1, 2) = A_i;
    B.block(edge_counter, 0, vOb.at(i) - 1, 1) = B_i;
    edge_counter += vOb[i] - 1;
  }

}

}  // namespace robot_plann