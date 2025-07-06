#!/bin/bash

# 编译
catkin_make

# 删除旧的文件
rm drl_moudle/ocp_planner_py.cpython-38-x86_64-linux-gnu.so

# 复制新的文件
cp devel/lib/ocp_planner_py.cpython-38-x86_64-linux-gnu.so drl_moudle/
