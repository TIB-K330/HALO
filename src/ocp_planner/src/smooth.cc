#include "smooth.h"

namespace robot_plann {
//
std::vector<SmoothPath> SmoothCorner::SmoothSharpCorner(
    const cv::Mat& costmap, std::vector<struct Point>& astar_path) {
  // Simplify the path by keeping only the two endpoints on each side of the common line.
  astar_path = SimplifyPath(astar_path);
  // Floyd path smoothing, performed twice to optimize into the best path
  for (int i = astar_path.size() - 1; i > 0; i--) {
    for (int j = 0; j < i - 1; j++) {
      // Determine if there is an obstacle between 2 points (no obstacle is true)
      if (CheckObstacle(costmap, astar_path[i], astar_path[j])) {
        for (int k = i - 1; k > j; k--) { 
          astar_path.erase(astar_path.begin() + k);  // Remove redundant inflection points
        }  
        i = j;
        break;
      }
    }
  }
  
  for (int i = astar_path.size() - 1; i > 0; i--) {
    for (int j = 0; j < i - 1; j++) {
      if (CheckObstacle(costmap, astar_path[i], astar_path[j])) {
        for(int k = i - 1; k > j; k--) {
          astar_path.erase(astar_path.begin() + k);
        }
        i = j;
        break;
      }
    }
  }

  smooth_path_.clear();
  // Only two points, store a straight line directly     
  if (astar_path.size() == 2) {
    smooth_path_.push_back(SmoothPath(astar_path[0].x, 
    astar_path[0].y, astar_path[1].x, astar_path[1].y));
  } else { // More than two points, need arc interpolation
    struct Point last_end, end_p;
    last_end.x = astar_path[0].x;
    last_end.y = astar_path[0].y;
    // interpolation
    for (int i = 1; i < (int)astar_path.size() - 1; i++) {
      struct Point vector_1, vector_2, vector_3;
      double vector_len;
      double delta = delta_max_;
    
      vector_1.x = last_end.x - astar_path[i].x;
      vector_1.y = last_end.y - astar_path[i].y;
      vector_len = EuclideanNorm(vector_1.x, vector_1.y);

      double l1 = vector_len;
      vector_1.x /= vector_len;
      vector_1.y /= vector_len;

      vector_2.x = astar_path[i + 1].x - astar_path[i].x;
      vector_2.y = astar_path[i + 1].y - astar_path[i].y;
      vector_len = EuclideanNorm(vector_2.x, vector_2.y);

      double l2 = vector_len;
      vector_2.x /= vector_len;
      vector_2.y /= vector_len;
      // Corner of the turn
      double beta = acos(vector_1.x * vector_2.x + vector_1.y * vector_2.y);
      struct Point start_p, center;
      // Calculate path direction error
      double d = delta * cos(beta / 2) / (1 - sin(beta / 2));
      d = std::min({d, epsilon_, l1 / 2, l2 / 2});
      // Calculate the error in the direction of the vertical path
      delta = d * (1 - sin(beta / 2)) / cos(beta / 2);
      // arc radius
      double r = d * tan(beta / 2);
      start_p.x = astar_path[i].x + d * vector_1.x;
      start_p.y = astar_path[i].y + d * vector_1.y;
      end_p.x = astar_path[i].x + d * vector_2.x;
      end_p.y = astar_path[i].y + d * vector_2.y;

      vector_3.x = vector_1.x + vector_2.x;
      vector_3.y = vector_1.y + vector_2.y;
      vector_len = EuclideanNorm(vector_3.x, vector_3.y);
      
      vector_3.x /= vector_len;
      vector_3.y /= vector_len;
      center.x = astar_path[i].x + (r + delta) * vector_3.x;
      center.y = astar_path[i].y + (r + delta) * vector_3.y;

      double start_angle = atan2(start_p.y - center.y, start_p.x - center.x);
      double end_angle = atan2(end_p.y - center.y, end_p.x - center.x);
      
      if (start_angle - end_angle > M_PI) start_angle -= 2 * M_PI;
      if (start_angle - end_angle < -M_PI) start_angle += 2 * M_PI;
    
      // In order, save the straight line segments first, then the arc segments 
      // If it's the last turning point, save the last straight line segment as well
      smooth_path_.push_back(SmoothPath(last_end.x, last_end.y, 
                                        start_p.x, start_p.y));
      
      smooth_path_.push_back(SmoothPath(center.x, center.y, r, 
                                        start_angle, end_angle));
      
      if (i == astar_path.size() - 2) {
        smooth_path_.push_back(SmoothPath(
            end_p.x, end_p.y, astar_path[i + 1].x, astar_path[i + 1].y));
      }
      last_end = end_p;
    }
  }
  return smooth_path_;
}

std::vector<struct Point> SmoothCorner::SimplifyPath(
    const std::vector<struct Point>& astar_path) {
  if (astar_path.size() <= 2) return astar_path;
  std::vector<struct Point> simple_path;
  double last_k = atan2(astar_path[1].y - astar_path[0].y, 
                        astar_path[1].x - astar_path[0].x);
        
  simple_path.push_back(astar_path[0]);
  for (int i = 2; i < astar_path.size(); i++) {
    double k = atan2(astar_path[i].y - astar_path[i - 1].y, 
                     astar_path[i].x - astar_path[i - 1].x);
    
    if (fabs(k - last_k) < 0.0001) continue;
    simple_path.push_back(astar_path[i - 1]);
    last_k = k;
  }
  simple_path.push_back(astar_path[astar_path.size() - 1]);
  return simple_path;
}

bool SmoothCorner::CheckObstacle(const cv::Mat& costmap, struct Point p1,
                                 struct Point p2) {
  double vector_x = (p1.x - p2.x);
  double vector_y = (p1.y - p2.y);
  double dis = std::sqrt(vector_x * vector_x + vector_y * vector_y);
  if (dis <= 0.4) return true;

  int point_num = dis / 0.2;
  bool is_free = true;
  for (int i = 1; i <= point_num; i++) {
    double x = p2.x + vector_x * i / (point_num + 1);
    double y = p2.y + vector_y * i / (point_num + 1);
    int pixel_x = (x * 20 + 0.5);
    int pixel_y = (costmap.rows - 1 - y * 20 + 0.5);
    if (pixel_y < 1) pixel_y = 1;
    if (pixel_y > costmap.rows - 2) pixel_y = costmap.rows - 2;
    if (255 - costmap.at<unsigned char>(pixel_y + 1, pixel_x) > 250 &&
        255 - costmap.at<unsigned char>(pixel_y, pixel_x) > 250 &&
        255 - costmap.at<unsigned char>(pixel_y - 1, pixel_x) > 250) {
      is_free = false; 
      break;
    }
  }   
  return is_free;
}
//
}  // namespace robot_plann