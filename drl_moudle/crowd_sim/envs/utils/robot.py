from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, FullState
from crowd_sim.envs import policy
from docutils.parsers.rst.directives.tables import ListTable
from crowd_sim.envs.utils.action import ActionDiff
import numpy as np

from crowd_sim.envs.utils.utils import theta_mod


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.rotation_constraint = getattr(config, section).rotation_constraint
        self.kinematics = getattr(config, section).kinematics
        self.trajectory = None
        self.use_rk2: bool = True
        # self.test_trajectory = None
        

    def act(self, ob, supervised=False):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        if hasattr(self.policy, 'run_solver'):
            action, action_index = self.policy.predict(state, supervised)
        else:
            action, action_index = self.policy.predict(state)

        return action, action_index

    def get_state(self, ob, as_tensors=True):
        if as_tensors:
            if self.policy is None:
                raise AttributeError('Policy attribute has to be set!')
            return self.policy.transform(JointState(self.get_full_state(), ob))
        else:
            return JointState(self.get_full_state(), ob)

    def cal_trajectory_euler(self, actions: np.ndarray):
        self.use_rk2 = False
        if self.trajectory is None:
            self.trajectory = list()
            state = self.get_full_state()
            self.trajectory.append(state)
            for i in range(len(actions) - 1):
                new_state = self.integral_position(state, ActionDiff(actions[i][0], actions[i][1]))
                state = new_state
                self.trajectory.append(state)

        else:
            self.trajectory[0] = self.get_full_state()
            for i in range(len(actions) - 1):
                self.trajectory[i + 1] = (
                    self.integral_position(self.trajectory[i], ActionDiff(actions[i][0], actions[i][1])))

    def cal_trajectory_rk2(self, actions: np.ndarray):
        if self.trajectory is None:
            self.trajectory = list()
            for x in self.integral_states_rk2(actions):
                state = FullState(x[0],
                                  x[1],
                                  x[3] - x[4] * self.radius,
                                  x[3] + x[4] * self.radius,
                                  self.radius,
                                  self.gx,
                                  self.gy,
                                  self.v_pref,
                                  x[2])
                self.trajectory.append(state)
        else:
            for i, x in enumerate(self.integral_states_rk2(actions)):
                state = FullState(x[0],
                                  x[1],
                                  x[3] - x[4] * self.radius,
                                  x[3] + x[4] * self.radius,
                                  self.radius,
                                  self.gx,
                                  self.gy,
                                  self.v_pref,
                                  x[2])
                # print("rk x: ", state.px, "rk y: ", state.py)
                self.trajectory[i] = state

        # self.trajectory = self.integral_states_rk2(actions)
                
    def integral_states_rk2(self, actions: np.ndarray):
        # print("**************************")

        state = self.get_full_state()
        ans = list()
        x = list()
        x.append(state.px)
        x.append(state.py)
        x.append(state.theta)
        x.append((state.vx + state.vy) * 0.5)
        x.append(0.5 * (state.vy - state.vx) / state.radius)
        ans.append(x)
        # print("rk x: ", x[0], "rk y: ", x[1])

        for i in range(len(actions) - 1):
            u = list()
            u.append((actions[i][0] + actions[i][1]) * 0.5)   # acc
            u.append(5.0 * (actions[i][1] - actions[i][0]) / 3.0)   # dr
            x_next = self.rk2_step(x, u)
            # print("rk x: ", x_next[0], "rk y: ", x_next[1])
            x = x_next
            ans.append(x)

        return ans

    def get_trajectory(self):
        # return [self.px], [self.py]
        if self.trajectory is None:
            return [self.px], [self.py]
        else:
            # if self.use_rk2:
            #     x_elements = [state[0] for state in self.trajectory]
            #     y_elements = [state[1] for state in self.trajectory]
            # else:
            x_elements = [state.px for state in self.trajectory]
            y_elements = [state.py for state in self.trajectory]
            return x_elements, y_elements

    def integral_position(self, state: FullState, action):
        self.check_validity(action)
        if self.kinematics == 'differential':
            left_acc = action.al
            right_acc = action.ar
            vel_left = state.vx + left_acc * self.time_step
            vel_right = state.vy + right_acc * self.time_step

            if np.abs(vel_left) > self.v_pref:
                vel_left = vel_left * self.v_pref / np.abs(vel_left)

            if np.abs(vel_right) > state.v_pref:
                vel_right = vel_right * state.v_pref / np.abs(vel_right)

            t_right = (vel_right - state.vy) / (right_acc + 1e-9)
            t_left = (vel_left - state.vx) / (left_acc + 1e-9)
            s_right = (vel_right + state.vy) * (0.5 * t_right) + vel_right * (self.time_step - t_right)
            s_left = (vel_left + state.vx) * (0.5 * t_left) + vel_left * (self.time_step - t_left)
            s = (s_right + s_left) * 0.5
            d_theta = (s_right - s_left) / (2 * state.radius)
            s_direction = theta_mod(state.theta + d_theta * 0.5)
            end_theta = theta_mod(state.theta + d_theta)
            end_robot_x = state.px + s * np.cos(s_direction)
            end_robot_y = state.py + s * np.sin(s_direction)
            px = end_robot_x
            py = end_robot_y
            # print("robot px1: ", px, "robot py1: ", py)
            # linear_vel = (vel_left + vel_right) / 2.0
            # vx = linear_vel * np.cos(end_theta)
            # vy = linear_vel * np.sin(end_theta)
            theta = end_theta
        # FullState(self.px, self.py, self.v_left, self.v_right, self.radius, self.gx, self.gy, self.v_pref,
        #           self.theta)
            return FullState(px, py, vel_left, vel_right, self.radius, self.gx, self.gy, self.v_pref, theta)
        else:
            print("=========error========")
            pass

    

    def rk2_step(self, x, u):
        
        def continuous_dynamic_R(x, u):
            # 动力学模型方程
            return np.array([
                x[3] * np.cos(x[2]),
                x[3] * np.sin(x[2]),
                x[4],
                u[0],
                u[1]
            ])
        
        k1 = continuous_dynamic_R(x, u)
        k2 = continuous_dynamic_R(x + self.time_step * k1, u)
        
        x_next = x + (self.time_step * 0.5) * (k1 + k2)
        
        return x_next
    