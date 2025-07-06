#ifndef ASTAR_H
#define ASTAR_H

#include <stddef.h>
#include <iostream>
#include <algorithm>

#include "utils.h"
#include "types.h"

namespace robot_plann {
//
struct AStarNode {
  int parent[2];
  int node_id;
  double f;
  double h;
  double g;
  int ex_cost;
  bool is_in_close_list;
  bool is_in_open_list;
};

class ListNode {
 public:
  ListNode(int x, int y, double cost) {
    x_ = x;
    y_ = y;
    cost_ = cost;
  }
  
  int x_;
  int y_;
  double cost_;
};

class AStar {
 public:
  AStar(int verbose): verbose_(verbose) {}
  ~AStar() {}
  void InitMap(const cv::Mat &costmap);
  std::vector<struct Point> SearchPath(const cv::Mat &costmap, int start_x, 
                                       int start_y, int goal_x, int goal_y);
  
  std::vector<struct Point> SearchPath(const cv::Mat &costmap,
                                       const Eigen::Vector2d &start, 
                                       const Eigen::Vector2d &goal);

 private:
  std::vector<std::vector<struct AStarNode>> nodes_;
  std::vector<std::vector<int>> GetNeighbors(int x, int y);
  int width_;
  int height_;

  int verbose_;
};
//
}  // namespace robot_plann

#endif