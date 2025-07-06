/**
  ******************************************************************************
  * @file    utils.h
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2021/12/27
  * @brief   general motion planning for MobiRo @ tib_k331
  ******************************************************************************
  * @attention
  *
  ******************************************************************************
  */
#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>


namespace robot_plann {
//
template <typename T>
inline T square(T x) {return x * x;}

template <typename T>
inline T myabs(T x) {return (x >= 0)?x:-x;}

template <typename T>
inline void UpperBound(T &x, T bound) {
  x = (x > bound)?bound:x;
}

template <typename T>
inline void LowerBound(T &x, T bound) {
  x = (x < bound)?bound:x;
}

template <typename T>
inline T EuclideanNorm(T dx, T dy) {
  return std::sqrt(square(dx) + square(dy));
}

template <typename T>    
inline T EuclideanNorm(const T *pt_a, const T *pt_b) {
  return EuclideanNorm(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]);
}

template <typename T>
inline void Unwrap(T &angle) {
  if (angle > M_PI) angle -= 2 * M_PI;
  if (angle < -M_PI) angle += 2 * M_PI;
}

inline double AccelDistance(double start_vel, double end_vel, double accel) {
  return myabs((square(start_vel) - square(end_vel)) / (2 * accel));
}

// Calculate if  point (x3, y3) lies in the left side of directed line segment from (x1, y1) to (x2, y2)
inline bool CounterClockwise(double x1, double y1, double x2, double y2,
                             double x3, double y3) {
  Eigen::Vector2d vec1 = {x2 - x1, y2 - y1};
  Eigen::Vector2d vec2 = {x3 - x1, y3 - y1};
  return (vec1.x() * vec2.y() - vec1.y() * vec2.x() > 0)? true: false;
}

// calculate if point (px, py) lies in the polygons represented by vertices (clockwise)
inline bool PointInPloy(double px, double py, 
                        const std::vector<Eigen::Vector2d> &vertices) {
  if (!vertices.size()) return false;

  for (int i = 0; i < (int)vertices.size() - 1; i++) {
    double p1_x = vertices[i].x();
    double p1_y = vertices[i].y();
    double p2_x = vertices[i + 1].x();
    double p2_y = vertices[i + 1].y();
    if (CounterClockwise(p1_x, p1_y, p2_x, p2_y, px, py)) return false;
  }

  double fx = vertices.front().x();
  double fy = vertices.front().y();
  double bx = vertices.back().x();
  double by = vertices.back().y();
  
  return CounterClockwise(bx, by, fx, fy, px, py)? false: true;
}
//
}

#endif  // UTILS_H