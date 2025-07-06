#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "planner.h"

namespace py = pybind11;

PYBIND11_MODULE(ocp_planner_py, m) {
    py::class_<robot_plann::Point>(m, "Point")
        .def(py::init<>())
        .def_readwrite("x", &robot_plann::Point::x)
        .def_readwrite("y", &robot_plann::Point::y)
        .def_readwrite("v", &robot_plann::Point::v);
    
    py::class_<robot_plann::Trajectory>(m, "Trajectory")
        .def(py::init<>());

    py::class_<robot_plann::ForPythonWall>(m, "ForPythonWall")
        .def(py::init<>())
        .def_readwrite("sx", &robot_plann::ForPythonWall::sx)
        .def_readwrite("sy", &robot_plann::ForPythonWall::sy)
        .def_readwrite("ex", &robot_plann::ForPythonWall::ex)
        .def_readwrite("ey", &robot_plann::ForPythonWall::ey);

    py::class_<robot_plann::RobotState>(m, "RobotState")
        .def(py::init<>())
        .def_readwrite("px", &robot_plann::RobotState::px)
        .def_readwrite("py", &robot_plann::RobotState::py)
        .def_readwrite("yaw", &robot_plann::RobotState::yaw)
        .def_readwrite("v", &robot_plann::RobotState::v)
        .def_readwrite("yaw_rate", &robot_plann::RobotState::yaw_rate)
        .def_readwrite("radius", &robot_plann::RobotState::radius)
        .def_readwrite("gx", &robot_plann::RobotState::gx)
        .def_readwrite("gy", &robot_plann::RobotState::gy)
        .def_readwrite("v_pref", &robot_plann::RobotState::v_pref); // 默认值会被自动处理

    // HumanState 
    py::class_<robot_plann::HumanState>(m, "HumanState")
        .def(py::init<>())
        .def_readwrite("px", &robot_plann::HumanState::px)
        .def_readwrite("py", &robot_plann::HumanState::py)
        .def_readwrite("vx", &robot_plann::HumanState::vx)
        .def_readwrite("vy", &robot_plann::HumanState::vy)
        .def_readwrite("radius", &robot_plann::HumanState::radius)
        .def_readwrite("pred_trajectorys", &robot_plann::HumanState::pred_trajectorys);

    // ObstacleState 
    py::class_<robot_plann::ObstacleState>(m, "ObstacleState")
        .def(py::init<>())
        .def_readwrite("px", &robot_plann::ObstacleState::px)
        .def_readwrite("py", &robot_plann::ObstacleState::py)
        .def_readwrite("radius", &robot_plann::ObstacleState::radius);

    // PolygonStateForPython 
    py::class_<robot_plann::PolygonStateForPython>(m, "PolygonStateForPython")
        .def(py::init<>())
        .def_readwrite("vertices", &robot_plann::PolygonStateForPython::vertices);

    // JointStateForPython 
    py::class_<robot_plann::JointStateForPython>(m, "JointStateForPython")
        .def(py::init<>())
        .def_readwrite("robot", &robot_plann::JointStateForPython::robot)
        .def_readwrite("hum", &robot_plann::JointStateForPython::hum)
        .def_readwrite("obst", &robot_plann::JointStateForPython::obst)
        .def_readwrite("rect", &robot_plann::JointStateForPython::rect)
        .def_readwrite("walls", &robot_plann::JointStateForPython::walls);

    // MPCInputForPython 
    py::class_<robot_plann::MPCInputForPython>(m, "MPCInputForPython")
        .def(py::init<>())
        .def_readwrite("ob", &robot_plann::MPCInputForPython::ob)
        .def_readwrite("sub_goal", &robot_plann::MPCInputForPython::sub_goal)
        .def_readwrite("valid", &robot_plann::MPCInputForPython::valid);

    // ControlVar 
    py::class_<robot_plann::ControlVar>(m, "ControlVar")
        .def(py::init<>())
        .def_readwrite("al", &robot_plann::ControlVar::al)
        .def_readwrite("ar", &robot_plann::ControlVar::ar);

    // MPCOutputForPython 
    py::class_<robot_plann::MPCOutputForPython>(m, "MPCOutputForPython")
        .def(py::init<>())
        .def_readwrite("success", &robot_plann::MPCOutputForPython::success)
        .def_readwrite("al", &robot_plann::MPCOutputForPython::al)
        .def_readwrite("ar", &robot_plann::MPCOutputForPython::ar)
        .def_readwrite("revised_goal", &robot_plann::MPCOutputForPython::revised_goal)
        .def_readwrite("astar_path", &robot_plann::MPCOutputForPython::astar_path)
        .def_readwrite("st_path", &robot_plann::MPCOutputForPython::st_path)
        .def_readwrite("control_vars", &robot_plann::MPCOutputForPython::control_vars);

    py::class_<robot_plann::Planner>(m, "OcpPlanner")
        .def(py::init<>())
        .def("run_mpc_solver", &robot_plann::Planner::RunSlover);

}



