import numpy as np
import socialforce
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, JointState_2tyeps


class SocialForce(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'SocialForce'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.initial_speed = 1
        self.v0 = 10
        self.sigma = 0.5
        self.sim = None

    def configure(self, config, device='cpu'):
        self.set_common_parameters(config)
        return

    def set_common_parameters(self, config):
        self.kinematics = config.action_space.kinematics
        self.sampling = config.action_space.sampling
        self.speed_samples = config.action_space.speed_samples
        self.rotation_samples = config.action_space.rotation_samples
        self.rotation_constraint = config.action_space.rotation_constraint

    def set_phase(self, phase):
        return

    def predict(self, state):
        """

        :param state:
        :return:
        """
        state = self.state_transform(state)
        sf_state = []
        self_state = state.robot_state
        cur_theta = self_state.theta
        sf_state.append((self_state.px, self_state.py, self_state.vx, self_state.vy, self_state.gx, self_state.gy))
        for human_state in state.human_states:
            # approximate desired direction with current velocity
            if human_state.vx == 0 and human_state.vy == 0:
                gx = human_state.px + (np.random.random() - 0.5) * self.time_step
                gy = human_state.py + (np.random.random() - 0.5) * self.time_step
            else:
                gx = human_state.px + human_state.vx
                gy = human_state.py + human_state.vy
            sf_state.append((human_state.px, human_state.py, human_state.vx, human_state.vy, gx, gy))
        sim = socialforce.Simulator(np.array(sf_state), delta_t=self.time_step, initial_speed=self.initial_speed,
                                    v0=self.v0, sigma=self.sigma)
        sim.step()
        if self.kinematics == 'holonomic':
            action = ActionXY(sim.state[0, 2], sim.state[0, 3])
        elif self.kinematics == 'unicycle':
            action = ActionXY(sim.state[0, 2], sim.state[0, 3])
            theta = np.arctan2(action.vy, action.vx)
            vel = np.sqrt(action.vx*action.vx + action.vy * action.vy)
            theta = (theta - cur_theta + np.pi) % (2 * np.pi) - np.pi
            action = ActionRot(vel, theta)
        # vel_index = (np.sqrt(action.vx * action.vx + action.vy * action.vy) // d_vel + 1) // 2
        # if vel_index > speed_samples:
        #     vel_index = speed_samples
        # d_rot = np.pi / rotation_samples
        # rot_index = (np.arctan2(action.vy, action.vx) // d_rot + 1) // 2
        # if rot_index < 0:
        #     rot_index = rot_index + rotation_samples
        # if vel_index == 0:
        #     action_index = int(0)
        # else:
        #     action_index = int((vel_index - 1) * rotation_samples + rot_index + 1)
        # action = ActionXY(vel_index * d_vel * 2.0 * np.cos(rot_index * d_rot * 2.0), vel_index * d_vel * 2.0 *
        #                   np.sin(rot_index * d_rot * 2.0))
        action_index = -1
        self.last_state = state

        return action, action_index

    def state_transform(self,state):
        human_state = state.human_states
        obstacle_state = state.obstacle_states
        robot_state = state.robot_state
        for i in range((len(obstacle_state))):
            obstacle_human = ObservableState(obstacle_state[i].px, obstacle_state[i].py, 0.0, 0.0, obstacle_state[i].radius)

            human_state.append(obstacle_human)
        state = JointState_2tyeps(robot_state, human_state)
        return state


class CentralizedSocialForce(SocialForce):
    """
    Centralized socialforce, a bit different from decentralized socialforce, where the goal position of other agents is
    set to be (0, 0)
    """
    def __init__(self):
        super().__init__()

    def predict(self, state):
        sf_state = []
        for agent_state in state:
            sf_state.append((agent_state.px, agent_state.py, agent_state.vx, agent_state.vy,
                             agent_state.gx, agent_state.gy))

        sim = socialforce.Simulator(np.array(sf_state), delta_t=self.time_step, initial_speed=self.initial_speed,
                                    v0=self.v0, sigma=self.sigma)
        sim.step()
        actions = [ActionXY(sim.state[i, 2], sim.state[i, 3]) for i in range(len(state))]
        del sim

        return actions
