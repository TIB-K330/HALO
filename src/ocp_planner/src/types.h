/**
  ******************************************************************************
  * @file    types.h
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2023/05/05
  * @brief   ocp planner as rl backend for MobiRo @ tib_k331
  ******************************************************************************
  * @attention
  *
  ******************************************************************************
  */
#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>
#include <vector>
#include <memory>


namespace robot_plann {

static constexpr float kMapResol = 0.05;
static constexpr float kHalfMapHeight = 10.0;
static constexpr float kHalfMapWidth = 6.0;

static constexpr float kVisualScale = 3.0;
static constexpr float kInflationRadius = 0.3;  // typically equivalent to the robot's radius
static constexpr int kNP = 10;
static constexpr float kDT = 0.25;

static constexpr double kMaxLinearVel  = 1.0;
static constexpr double kMaxLinearAcc  = 1.0;
static constexpr double kMaxAngularVel = 3.0;
static constexpr double kMaxAngularAcc = 3.0;

// st search
static constexpr double kSTHalfLocalRange = kDT * (kNP - 1) * 1.0;
static constexpr double kSTHalfLocalRangeSquare = 
                        kSTHalfLocalRange * kSTHalfLocalRange;

static constexpr std::size_t kThreadNum = 8;

struct Point {
  double x;
  double y;
  double v;
};

struct RobotState {
  double px;
  double py;
  double yaw;
  double v;
  double yaw_rate;

  double radius;
  double gx;
  double gy;
  double v_pref = 1.0;
};

typedef std::vector<Point> Trajectory;
typedef std::pair<Point, Point> Wall;

struct ForPythonWall {
  double sx;
  double sy;
  double ex;
  double ey;
};


struct HumanState {
  double px;
  double py;
  double vx;
  double vy;
  double radius;
  // Temp: Consider point.v as a predictive probability
  std::vector<Trajectory> pred_trajectorys; 
  // std::vector<double> prob_pred_trajectorys;

};

struct ObstacleState {
  double px;
  double py;
  double radius;
};

struct PolygonState {
  std::vector<Eigen::Vector2d> vertices;
};

struct PolygonStateForPython {
  std::vector<Point> vertices;
};

struct JointState {
  RobotState robot;
  std::vector<HumanState> hum;
  std::vector<ObstacleState> obst;
  PolygonState rect;
  std::vector<Wall> walls;
};

struct JointStateForPython {
  RobotState robot;
  std::vector<HumanState> hum;
  std::vector<ObstacleState> obst;
  PolygonStateForPython rect;
  std::vector<ForPythonWall> walls;
};

struct MpcParams {
  uint16_t np;
  double dt;
  double max_linear_vel;
  double max_angular_vel;
  double max_linear_acc;
  double max_angular_acc;
  int local_obst_num;

  typedef std::shared_ptr<MpcParams> Ptr;
};

struct State {
  double X;
  double Y;
  double phi;
  double vx;
  double r;  // yaw rate
  std::vector<double> lambda;
  std::vector<double> mu;

  void setZero(int edge_num, int obst_num) {
    X   = 0.0;
    Y   = 0.0;
    phi = 0.0;
    vx  = 0.0;
    r   = 0.0;
    
    lambda.resize(edge_num);
    for (int i = 0; i < edge_num; i++) lambda[i] = 0.0;

    mu.resize(obst_num * 4);
    for (int i = 0; i < obst_num * 4; i++) mu[i] = 0.0;
  }

  void unwrap() {
    if (phi > M_PI) phi -= 2.0 * M_PI;
    if (phi < -M_PI) phi += 2.0 * M_PI;
  }
};

struct Input {
  double acc;
  double dr;

  void setZero() {
    acc = 0.0;
    dr  = 0.0;
  }
};

struct OptVariables {
  State xk;
  Input uk;
};

struct OptVarIndex {
  int state_var_num;
  int obst_dual_num;
  int shape_dual_num;
  int input_var_num;
};

typedef std::array<robot_plann::OptVariables, kNP> MpcStages;
// typedef std::vector<State> Path;
struct Position {
  Point position;
};

struct Pose {
  Position pose;
};
struct Path {
  std::vector<Pose> poses;
};

struct MpcReturn {
  MpcStages stages;
  bool success;
};

struct HyperPlane {
  double nx;
  double ny;
  double b;
};

struct DynaObstacle {
  double id;
  double distance;
  double radius;
  std::vector<Trajectory> pred_trajectorys;
};

struct ObstacleArray {
  std::vector<DynaObstacle> dyna_obstacles{};
};

// for python input and out put

struct MPCInputForPython {
  JointStateForPython ob;
  Point sub_goal;
  bool valid = false;
};

struct ControlVar {
  double al;
  double ar;
};
struct MPCOutputForPython {
  bool success{false};
  double al = 0.0;
  double ar = 0.0;

  Point revised_goal{};
  std::vector<Point> astar_path{};
  std::vector<Point> st_path{};
  std::vector<ControlVar> control_vars{};
};

}  // namespace robot_plann

#endif  // TYPES_H