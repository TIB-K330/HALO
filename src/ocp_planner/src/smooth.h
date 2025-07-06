#ifndef SMOOTH_H
#define SMOOTH_H

#include "astar.h"

namespace robot_plann {
//
class SmoothPath { 
 public:
  SmoothPath(double start_x, double start_y, double end_x, double end_y) {
    start_x_ = start_x;
    start_y_ = start_y;
    end_x_ = end_x;
    end_y_ = end_y;
    length_ = EuclideanNorm(start_x_ - end_x_, start_y_ - end_y_);
    is_arc_ = false;
  }
  
  SmoothPath(double center_x, double center_y, double r, double start_angle, 
              double end_angle) {
    center_x_ = center_x;
    center_y_ = center_y;
    radius_ = r;
    start_angle_ = start_angle;
    end_angle_ = end_angle;
    double delta_angle = end_angle - start_angle;
    if (delta_angle > M_PI) delta_angle -= 2 * M_PI;
    if (delta_angle < -M_PI) delta_angle += 2 * M_PI;
    length_ = r * fabs(delta_angle);  // arc
    is_arc_ = true;
  }
  
  ~SmoothPath() = default;
        
  double start_x_;
  double start_y_;
  double end_x_;
  double end_y_;
  double length_;

  double center_x_;
  double center_y_;
  double start_angle_;
  double end_angle_;
  double radius_;

  double start_vel_;
  double end_vel_;

  bool is_arc_;
};


class SmoothCorner {
 public:
  inline void SetDeltaMax(double delta_max) {delta_max_ = delta_max;}

  std::vector<SmoothPath> SmoothSharpCorner(
      const cv::Mat& costmap, std::vector<struct Point>& astar_path);

 private:
  std::vector<struct Point> SimplifyPath(
      const std::vector<struct Point>& astar_path);
  
  bool CheckObstacle(const cv::Mat& costmap, struct Point p1, struct Point p2);
  
  std::vector<SmoothPath> smooth_path_;
  double delta_max_;
  double epsilon_ = 0.4;
};
//
}  // namespace robot_plann

#endif  // SMOOTH_H