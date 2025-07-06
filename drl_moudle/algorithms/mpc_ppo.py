import math
import time
from collections import deque
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union, List
import gym
import numpy as np
import torch as th
from sympy import false
from sympy.physics.units import action
from torch.ao.quantization.backend_config.native import input_output_only_quint8_dtype_config
# from crowd_nav.utils.rvo_test import robot_state
from torch.nn import functional as F
import threading
import queue

import crowd_sim.envs
from .graph_ppo import (
    ActionDiff,
    BaseCallback,
    configure_logger,
    GraphActorCriticPolicy,
    GraphPPO,
    JointState,
    joint_state_as_graph,
    MaybeCallback,
    ReachGoal,
    safe_mean,
    Schedule,
    Timeout,
)
from modules.mpc_ppo_core import MaskedActorCriticPolicy, MaskedRolloutBuffer
from modules.utils import explained_variance, get_schedule_fn
# from crowd_sim.envs.utils.utils import map_action_to_goal
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.state import  FullState

import ocp_planner_py

# ros modules
import rospy
from ocp_planner.srv import OcpLocalPlann, OcpLocalPlannRequest
from ocp_planner.msg import HumanState as HumanStateMsg
from ocp_planner.msg import ObstacleState as ObstacleStateMsg
from ocp_planner.msg import PolyState, Point, WallState
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped 
from std_msgs.msg import Header

class MpcPPO(GraphPPO):
    policy_aliases: ClassVar[Dict[str, Any]] = {"GraphPolicy": MaskedActorCriticPolicy, }

    def __init__(
        self,
        policy: Union[str, Type[GraphActorCriticPolicy]],
        env: gym.Env,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Union[None, float] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        target_kl: Optional[float] = None,
        policy_kwargs: Optional[Dict[str, any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        action_dim: int = 9,
        goal_coord_range: Tuple[float, float] = (-2.0, -2.0),
        use_ros: bool = False,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            target_kl=target_kl,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
        )
        self.use_ros = use_ros

        if self.use_ros:
            self.ocp_planner = rospy.ServiceProxy('/ocp_plann', OcpLocalPlann)
        else:
            self.ocp_planner = ocp_planner_py.OcpPlanner()
            # self.ocp_planner_thread = ocp_planner_py.ThreadOcpPlanner()

        self._raw_last_state: Optional[JointState] = None
        self.last_local_goal: Optional[Tuple[float, float]] = None
        self.MPC_NP = 10
        self.action_index: int = 0
        self.action_prob: np.ndarray = np.zeros(action_dim * action_dim)
        self.num_actions_per_dim: int = action_dim
        self.goal_coord_range: Tuple[float, float] = goal_coord_range

    def _setup_model(self) -> None:
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        self.set_random_seed(self.seed)
        self.rollout_buffer = MaskedRolloutBuffer(
            self.n_steps,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        # The observation of crowd_sim is not compliance with the gym standard
        self.policy = self.policy_class(self.action_space, self.lr_schedule, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def run_solver_ros(self, observation: JointState, local_goal: Tuple[float, float]) -> np.ndarray:
        req = OcpLocalPlannRequest()
        req.ob.robot_state.pose.x = observation.robot_state.px
        req.ob.robot_state.pose.y = observation.robot_state.py
        req.ob.robot_state.pose.theta = self.wrap_pi(observation.robot_state.theta)
        req.ob.robot_state.vl = observation.robot_state.vx
        req.ob.robot_state.vr = observation.robot_state.vy
        req.ob.robot_state.radius = observation.robot_state.radius
        req.ob.robot_state.gx = observation.robot_state.gx
        req.ob.robot_state.gy = observation.robot_state.gy

        if len(observation.human_states) == len(observation.human_pred_states):
            for human, pred_path in zip(observation.human_states, observation.human_pred_states):
                human_state = HumanStateMsg()
                human_state.px = human.px
                human_state.py = human.py
                human_state.vx = human.vx
                human_state.vy = human.vy
                human_state.radius = human.radius

                human_pred_path = list()
                trajectory = NavPath()
                trajectory.header = Header()
                for x, y in zip(pred_path[0], pred_path[1]):
                    pose = PoseStamped()
                    pose.header = trajectory.header
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    pose.pose.position.z = 1.0
                    pose.pose.orientation.x = 0.0
                    pose.pose.orientation.y = 0.0
                    pose.pose.orientation.z = 0.0
                    pose.pose.orientation.w = 1.0
                    trajectory.poses.append(pose)

                human_pred_path.append(trajectory)
                human_state.trajectory = human_pred_path
                # for
                req.ob.human_states.append(human_state)
        else:
            for hum in observation.human_states:
                hum_msg = HumanStateMsg(hum.px, hum.py, hum.vx, hum.vy, hum.radius, [])
                req.ob.human_states.append(hum_msg)

        for obst in observation.obstacle_states:
            obst_msg = ObstacleStateMsg(obst.px, obst.py, obst.radius)
            req.ob.obstacle_states.append(obst_msg)

        # ignore the front edges that do NOT belong to the central polygon obstacle
        if len(observation.wall_states) > 2:
            poly_edges = observation.wall_states[2:]
            poly_state = PolyState()

            for edge in poly_edges:
                pt = Point()
                pt.x = edge.sx
                pt.y = edge.sy
                poly_state.vertices.append(pt)

            poly_state.vertices = poly_state.vertices[::-1]  # ccw2cw
            poly_state.is_clockwise = True
            req.ob.poly_states.append(poly_state)

        # walls left_down to right_up
        left_wall_state = WallState()
        left_wall = observation.wall_states[0]
        left_wall_state.sx = left_wall.sx + observation.robot_state.radius + 0.05
        left_wall_state.sy = left_wall.sy
        left_wall_state.ex = left_wall.ex + observation.robot_state.radius + 0.05
        left_wall_state.ey = left_wall.ey
        req.ob.walls.append(left_wall_state)
        right_wall_state = WallState()
        right_wall = observation.wall_states[1]
        right_wall_state.sx = right_wall.sx - (observation.robot_state.radius + 0.05)
        right_wall_state.sy = right_wall.sy
        right_wall_state.ex = right_wall.ex - (observation.robot_state.radius + 0.05)
        right_wall_state.ey = right_wall.ey
        req.ob.walls.append(right_wall_state)
        # trans the local goal in robot-centric frame to map frame
        robot_pose = (observation.robot_state.px, observation.robot_state.py, observation.robot_state.theta)
        self.last_local_goal = self.robot2map(local_goal, robot_pose)  # save for visual
        req.sub_goal.x = self.last_local_goal[0]
        req.sub_goal.y = self.last_local_goal[1]

        res = self.ocp_planner.call(req)  # call ocp_planner to plan a local optimal trajectory
        if len(res.astar_path) != 0:
            astar_path = np.zeros([len(res.astar_path), 3])
            for i, pt in enumerate(res.astar_path):
                astar_path[i, 0] = pt.x
                astar_path[i, 1] = pt.y
            self.env.set_astar_path(astar_path)
        else:
            print("astar search fail")

        control_vars = np.zeros((10, 2), dtype=float)
        if len(res.control_vars) != 0:
            for i, var in enumerate(res.control_vars):
                control_vars[i][0] = var.al
                control_vars[i][1] = var.ar

        self.env.robot.cal_trajectory_rk2(control_vars)

        return np.array((res.al, res.ar)), control_vars, res.success

    def run_solver(self, observation: JointState, local_goal: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, bool]:

        input = self.get_pybind_input(observation, local_goal)
        ans = self.ocp_planner.run_mpc_solver(input)  # call ocp_planner to plan a local optimal trajectory

        action, control_vars, success = self.handle_pybind_output(ans)
        return action, control_vars, success

    def handle_pybind_output(self, ans) -> Tuple[np.ndarray, np.ndarray, bool]:
        if len(ans.astar_path) != 0:
            astar_path = np.zeros([len(ans.astar_path), 3])
            for i, pt in enumerate(ans.astar_path):
                astar_path[i, 0] = pt.x
                astar_path[i, 1] = pt.y
            self.env.set_astar_path(astar_path)

        control_vars = np.zeros((10, 2), dtype=float)
        if len(ans.control_vars) != 0:
            for i, var in enumerate(ans.control_vars):
                control_vars[i][0] = var.al
                control_vars[i][1] = var.ar

        self.env.robot.cal_trajectory_rk2(control_vars)

        return np.array((ans.al, ans.ar)), control_vars, ans.success

    def get_pybind_input(self, observation: JointState, local_goal: Tuple[float, float]):
        input = ocp_planner_py.MPCInputForPython()

        input.ob.robot.gx = observation.robot_state.gx
        input.ob.robot.gy = observation.robot_state.gy
        input.ob.robot.px = observation.robot_state.px
        input.ob.robot.py = observation.robot_state.py
        input.ob.robot.v = 0.5 * (observation.robot_state.vx + observation.robot_state.vy)
        input.ob.robot.yaw = self.wrap_pi(observation.robot_state.theta)
        input.ob.robot.radius = observation.robot_state.radius
        input.ob.robot.yaw_rate = (
                0.5 * (observation.robot_state.vy - observation.robot_state.vx) / observation.robot_state.radius)


        hums = list()
        if len(observation.human_states) == len(observation.human_pred_states):

            for human, pred_path in zip(observation.human_states, observation.human_pred_states):
                human_state = ocp_planner_py.HumanState()
                human_state.px = human.px
                human_state.py = human.py
                human_state.vx = human.vx
                human_state.vy = human.vy
                human_state.radius = human.radius

                human_pred_path = list()
                trajectory = list()
                for x, y in zip(pred_path[0], pred_path[1]):
                    pt = ocp_planner_py.Point()
                    pt.x = x
                    pt.y = y
                    pt.v = 1.0

                    trajectory.append(pt)

                human_pred_path.append(trajectory)
                human_state.pred_trajectorys = human_pred_path
                hums.append(human_state)
        else:
            for human in observation.human_states:
                human_state = ocp_planner_py.HumanState()
                human_state.px = human.px
                human_state.py = human.py
                human_state.vx = human.vx
                human_state.vy = human.vy
                human_state.radius = human.radius

                hums.append(human_state)

        input.ob.hum = hums

        obst = list()
        for obstacle in observation.obstacle_states:
            obstacle_state = ocp_planner_py.ObstacleState()
            obstacle_state.px = obstacle.px
            obstacle_state.py = obstacle.py
            obstacle_state.radius = obstacle.radius
            obst.append(obstacle_state)
        input.ob.obst = obst

        vertices = list()
        # ignore the front edges that do NOT belong to the centeral polygon obstacle
        if len(observation.wall_states) > 2:
            poly_edges = observation.wall_states[2:]

            for edge in poly_edges:
                pt = ocp_planner_py.Point()
                pt.x = edge.sx
                pt.y = edge.sy
                pt.v = 0.0
                vertices.append(pt)
        vertices = vertices[::-1]  # ccw2cw
        input.ob.rect.vertices = vertices

        walls = list()
        # walls left_down to right_up
        left_wall_state = ocp_planner_py.ForPythonWall()
        left_wall = observation.wall_states[0]
        left_wall_state.sx = left_wall.sx + observation.robot_state.radius + 0.05
        left_wall_state.sy = left_wall.sy
        left_wall_state.ex = left_wall.ex + observation.robot_state.radius + 0.05
        left_wall_state.ey = left_wall.ey
        walls.append(left_wall_state)
        right_wall_state = ocp_planner_py.ForPythonWall()
        right_wall = observation.wall_states[1]
        right_wall_state.sx = right_wall.sx - (observation.robot_state.radius + 0.05)
        right_wall_state.sy = right_wall.sy
        right_wall_state.ex = right_wall.ex - (observation.robot_state.radius + 0.05)
        right_wall_state.ey = right_wall.ey
        walls.append(right_wall_state)

        input.ob.walls = walls

        robot_pose = (observation.robot_state.px, observation.robot_state.py, observation.robot_state.theta)
        
        self.last_local_goal = self.robot2map(local_goal, robot_pose)  # save for visual

        input.sub_goal.x = self.last_local_goal[0]
        input.sub_goal.y = self.last_local_goal[1]
        input.sub_goal.v = 0.0

        input.valid = True

        return input

    def collect_rollouts(
        self,
        env: crowd_sim.envs.CrowdSim,
        callback: BaseCallback,
        rollout_buffer: MaskedRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_state is not None, "No previous state graph was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            action_mask = env.action_mask

            with th.no_grad():
                actions, values, log_probs = self.policy(self._last_state, action_mask=action_mask)
                self.action_prob = self.policy.actions_probs.cpu().numpy()[0]

            actions = actions.cpu().numpy()
            env.set_action_index(actions[0])
            env.set_actions_prob(self.action_prob)

            if self.use_ros:
                wheel_acc, control_vars, opt_success = (
                    self.run_solver_ros(self._raw_last_state, self.map_action_to_goal(action_index=actions[0])))
            else:
                wheel_acc, control_vars, opt_success = self.run_solver(self._raw_last_state,
                                                                       self.map_action_to_goal(action_index=actions[0]))

            # env.render(mode='debug', local_goal_vis=self.last_local_goal)
            new_ob, reward, done, info = env.step(ActionDiff(wheel_acc[0], wheel_acc[1]))
            new_robot_state = env.robot.get_full_state()

            self.num_timesteps += 1

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self.rewards.append(float(reward))  # for return calculation
            if done:
                returns = []
                for step in range(len(self.rewards)):
                    step_return = sum([pow(self.gamma, t) * reward for t, reward in enumerate(self.rewards[step:])])
                    returns.append(step_return)

                ep_info = {"r": round(safe_mean(returns), 6), "l": len(self.rewards)}
                self.rewards = []

                self.ep_info_buffer.append(ep_info)
                self.ep_success_buffer.append(True if isinstance(info, ReachGoal) else False)

            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            if done:
                if isinstance(info, Timeout):
                    terminal_state = joint_state_as_graph(JointState(new_robot_state, new_ob), device=self.device)
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_state).cpu().numpy()[0]
                    reward += self.gamma * terminal_value

                self._episode_num += 1

                new_ob = env.reset(phase="train")
                new_robot_state = env.robot.get_full_state()

            rollout_buffer.add(
                self._last_state,
                actions,
                reward,
                self._last_episode_start,
                values,
                log_probs,
                action_mask,
            )
            self._raw_last_state = JointState(new_robot_state, new_ob)  # for ocp planner

            self._last_state = joint_state_as_graph(self._raw_last_state, device=self.device)  # for neural network
            self._last_episode_start = done

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(self._last_state)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=done)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.policy.set_training_mode(True)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        # save data for logging or visualization
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True  # early stop flag
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batch size == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                # an introduction to loss https://zhuanlan.zhihu.com/p/679147320
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            total_timesteps += self.num_timesteps  # Make sure training timesteps are ahead of the internal counter

        if reset_num_timesteps or self._last_state is None:
            assert self.env is not None
            ob = self.env.reset(phase="train")
            robot_state = self.env.robot.get_full_state()
            self._raw_last_state = JointState(robot_state, ob)  # for ocp_planner
            self._last_state = joint_state_as_graph(self._raw_last_state, device=self.device)
            self._last_episode_start = True  # Original SB3 is a ndarray, for multiple environment

        if not self._custom_logger:
            self._logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
        # Create eval callback if needed
        callback = self._init_callback(callback)
        return total_timesteps, callback

    def predict(
        self,
        observation: JointState,
        deterministic: bool = False,
        action_mask: Optional[th.Tensor] = None,
    ) -> np.ndarray:
        ob_graph = joint_state_as_graph(observation, device=self.device)
        action = self.policy.predict(ob_graph, action_mask=action_mask, 
                                     deterministic=deterministic).squeeze()
        self.action_prob = self.policy.actions_probs.cpu().numpy()[0]
        if deterministic:
            self.action_index = action
        if self.use_ros:
            ocp_output, _, _ = self.run_solver_ros(observation, self.map_action_to_goal(action))
        else:
        
            
            ocp_output, _, _ = self.run_solver(observation, self.map_action_to_goal(action))
        return np.expand_dims(ocp_output, axis=0)  # add the batch dim

    def get_action_index(self):
        return np.copy(self.action_index)

    def get_action_prob(self):
        return np.copy(self.action_prob)

    def map_action_to_goal(self, action_index: int) -> Tuple[float, float]:
        min_goal_coord, max_goal_coord = self.goal_coord_range
        action_values = np.linspace(min_goal_coord, max_goal_coord, self.num_actions_per_dim)
        x_index = action_index // self.num_actions_per_dim
        y_index = action_index % self.num_actions_per_dim

        goal_x = action_values[x_index]
        goal_y = action_values[y_index]

        return goal_x, goal_y
    @staticmethod
    def wrap_pi(x):
        x = (x + np.pi) % (2 * np.pi)
        if x < 0:
            x = x + 2 * np.pi
        return x - np.pi

    @staticmethod
    def robot2map(pt: Tuple[float, float], robot_pose: Tuple[float, float, float]) -> Tuple[float, float]:
        rotation_matrix = np.array([[np.cos(robot_pose[2]), -np.sin(robot_pose[2])],
                                    [np.sin(robot_pose[2]), np.cos(robot_pose[2])]])

        trans_vec = np.array([[robot_pose[0]], [robot_pose[1]]])
        pt_in_map = np.dot(rotation_matrix, np.array([[pt[0]], [pt[1]]])) + trans_vec
        return pt_in_map[0][0], pt_in_map[1][0]
