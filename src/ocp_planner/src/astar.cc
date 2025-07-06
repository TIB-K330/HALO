#include "astar.h"

namespace robot_plann {

struct Compare {
  bool operator()(const ListNode& a, const ListNode& b) {
    return a.cost_ >= b.cost_;
  }
};

void AStar::InitMap(const cv::Mat& costmap) {
  width_ = costmap.cols;
  height_ = costmap.rows;
  nodes_.resize(costmap.cols);
  for (int i = 0; i < (int)nodes_.size(); i++) {
    nodes_[i].resize(costmap.rows);
    for(int j = 0; j < (int)nodes_[i].size(); j++) {
      nodes_[i][j].parent[0] = -1;
      nodes_[i][j].parent[1] = -1;
      nodes_[i][j].f = 0;
      nodes_[i][j].g = 0;
      nodes_[i][j].h = 0;
      nodes_[i][j].ex_cost = (255 - costmap.at<unsigned char>(j, i));
      nodes_[i][j].is_in_close_list = false;
      nodes_[i][j].is_in_open_list = false;
    }
  }
}

std::vector<struct Point> AStar::SearchPath(const cv::Mat& costmap,
                                            int start_x, int start_y, 
                                            int goal_x, int goal_y) {
  this->InitMap(costmap);
  start_y = costmap.rows - 1 - start_y;
  goal_y = costmap.rows - 1 - goal_y;

  std::vector<struct Point> path;
  // init astar open list 
  std::priority_queue<ListNode, std::vector<ListNode>, Compare> open_list;
  open_list.push(ListNode(start_x, start_y, nodes_[start_x][start_y].f));
  nodes_[start_x][start_y].is_in_open_list = true;
  // start a star graph search
  while (true) {  
    /* add the top node in open list to close list */
    int current_node_x = open_list.top().x_;
    int current_node_y = open_list.top().y_;
    nodes_[current_node_x][current_node_y].is_in_close_list = true;

    /* if current node is goal, then plann success and break */
    if (current_node_x == goal_x && current_node_y == goal_y)
      break;
            
    /* if current node is not goal and open list is empty, 
      then plann is failed, else remove current node from open list */
    if (!open_list.size()) {
      if (verbose_ >= 1) {
        std::cerr << "No astar solution found!!! " << std::endl;
      }
      return path;
    }
    open_list.pop();
    /* get all the neighbors of current point */
    std::vector<std::vector<int>> neighbors = GetNeighbors(current_node_x, 
                                                           current_node_y);
    if (!neighbors.size()) continue;
    /* iter over all neighbor pts of current point */
    AStarNode *neighbor_node = nullptr;

    for (auto iter = neighbors.begin(); iter != neighbors.end(); iter++) {
      /* euclidean distance as cost */
      double delta_x = iter->at(0) - current_node_x;
      double delta_y = iter->at(1) - current_node_y;
      double step_cost = EuclideanNorm(delta_x, delta_y);
      double cost_g = nodes_[current_node_x][current_node_y].g +
                      nodes_[current_node_x][current_node_y].ex_cost + 
                      step_cost;
      // use a pointer to reduce code length only          
      neighbor_node = &nodes_[iter->at(0)][iter->at(1)];
      // update existing node
      if (neighbor_node->is_in_open_list == true) {
        if (cost_g > neighbor_node->g) continue;
        /* update the exist pt in open list */
        neighbor_node->parent[0] = current_node_x;
        neighbor_node->parent[1] = current_node_y;
        neighbor_node->g = cost_g;
        neighbor_node->f = neighbor_node->g + neighbor_node->h;
      } else {  // add new point to the open list
        delta_x = fabs(goal_x - iter->at(0));
        delta_y = fabs(goal_y - iter->at(1));
        neighbor_node->parent[0] = current_node_x;
        neighbor_node->parent[1] = current_node_y;            
        neighbor_node->h = EuclideanNorm(delta_x, delta_y);
        neighbor_node->g = cost_g;
        neighbor_node->f = neighbor_node->g + neighbor_node->h;
        open_list.push(ListNode(iter->at(0), iter->at(1), neighbor_node->f));
        neighbor_node->is_in_open_list = true;
      } 
    }                                            
  }

  int current_x = goal_x;
  int current_y = goal_y;
  // traceback to obtain the optimal path
  while (true) {
    struct Point p;
    p.x = (double)current_x * kMapResol;
    p.y = (double)(height_ - 1 - current_y) * kMapResol;
    path.push_back(p);
    int parent_x = nodes_[current_x][current_y].parent[0];
    int parent_y = nodes_[current_x][current_y].parent[1];
    current_x = parent_x;
    current_y = parent_y;
    if (parent_x == start_x && parent_y == start_y) {
      struct Point p1;
      p1.x = (double)parent_x * kMapResol;
      p1.y = (double)(height_ - 1 - parent_y) * kMapResol;
      path.push_back(p1);
      break;
    }
  }
  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<struct Point> AStar::SearchPath(const cv::Mat &costmap,
                                            const Eigen::Vector2d &start, 
                                            const Eigen::Vector2d &goal) {
  /*
      The coord system used in geometric plann has its origin at the
    bottom-left corner, with y axis increasing upwards and x increasing 
    to the right, which is different from cv::Mat.
  */
  int idx_sx = (int)round((start(0) + kHalfMapWidth) / kMapResol);
  int idx_sy = (int)round((start(1) + kHalfMapHeight) / kMapResol);
  
  int idx_gx = (int)round((goal(0) + kHalfMapWidth) / kMapResol);
  int idx_gy = (int)round((goal(1) + kHalfMapHeight) / kMapResol);

  if (idx_sy >= costmap.rows) idx_sy = costmap.rows - 1;
  if (idx_gy >= costmap.rows) idx_gy = costmap.rows - 1;
  
  if (verbose_ > 1) {
    std::cout << "DEBUG: (" << idx_sx << ", " << idx_sy << "), (" << 
        idx_gx << ", " << idx_gy << ")" << std::endl;
  }

  return SearchPath(costmap, idx_sx, idx_sy, idx_gx, idx_gy);
}

std::vector<std::vector<int>> AStar::GetNeighbors(int x, int y) {
  std::vector<std::vector<int>> neighbors;
  if (x > 0 && y > 0) {
    if (nodes_[x - 1][y - 1].is_in_close_list == false && 
        nodes_[x - 1][y - 1].ex_cost < 250)
      neighbors.push_back({x - 1, y - 1});
  }
  if (x > 0) {
    if (nodes_[x - 1][y].is_in_close_list == false && 
        nodes_[x - 1][y].ex_cost < 250)
      neighbors.push_back({x - 1, y});
  }
  if (x > 0 && y < height_ - 1) {
    if (nodes_[x - 1][y + 1].is_in_close_list == false && 
        nodes_[x - 1][y + 1].ex_cost < 250)
      neighbors.push_back({x - 1, y + 1});
    }
  if (y < height_ - 1) {
    if (nodes_[x][y + 1].is_in_close_list == false && 
        nodes_[x][y + 1].ex_cost < 250)
      neighbors.push_back({x, y + 1});
  }
  if (x < width_ - 1 && y < height_ - 1) {
    if (nodes_[x + 1][y + 1].is_in_close_list == false && 
        nodes_[x + 1][y + 1].ex_cost < 250)
      neighbors.push_back({x + 1, y + 1});
    }
  if (x < width_ - 1) {
    if (nodes_[x + 1][y].is_in_close_list == false && 
        nodes_[x + 1][y].ex_cost < 250)
      neighbors.push_back({x + 1, y});
  }
  if (x < width_ - 1 && y > 0) {
    if (nodes_[x + 1][y - 1].is_in_close_list == false && 
        nodes_[x + 1][y - 1].ex_cost < 250)
      neighbors.push_back({x + 1, y - 1});
  }
  if (y > 0) {
    if (nodes_[x][y - 1].is_in_close_list == false && 
        nodes_[x][y - 1].ex_cost < 250)
      neighbors.push_back({x, y - 1});
  }
  
  return neighbors;
}
//
}  // namespace robot_plann
