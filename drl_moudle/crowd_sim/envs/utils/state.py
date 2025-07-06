import torch


class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        # for differential model, vx and vy represent v_left and v_right, respectively.
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])

    def to_tuple(self):
        return self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

    def to_tuple(self):
        return self.px, self.py, self.vx, self.vy, self.radius


class ObstacleState(object):
    def __init__(self, px, py, radius):
        self.px = px
        self.py = py
        self.radius = radius
        self.position = (self.px, self.py)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.radius]])

    def to_tuple(self):
        return self.px, self.py, self.radius


class WallState(object):
    def __init__(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey

    def __str__(self):
        return ' '.join([str(x) for x in [self.sx, self.sy, self.ex, self.ey]])

    def to_tuple(self):
        return self.sx, self.sy, self.ex, self.ey


class JointState(object):
    def __init__(self, robot_state, observed_states):
        assert isinstance(robot_state, FullState)
        human_states = observed_states[0]
        obstacle_states = observed_states[1]
        wall_states = observed_states[2]

        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        for obstacle_state in obstacle_states:
            assert isinstance(obstacle_state, ObstacleState)

        for wall_state in wall_states:
            assert isinstance(wall_state, WallState)

        self.robot_state = robot_state
        self.human_states = human_states
        self.obstacle_states = obstacle_states
        self.wall_states = wall_states
        self.human_pred_states = observed_states[3]

    def to_tensor(self, add_batch_size=False, device=None):
        robot_state_tensor = torch.Tensor([self.robot_state.to_tuple()])
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in self.human_states])
        obstacle_states_tensor = torch.Tensor([obstacle_state.to_tuple() for obstacle_state in self.obstacle_states])
        wall_states_tensor = torch.Tensor([wall_state.to_tuple() for wall_state in self.wall_states])

        if add_batch_size:
            robot_state_tensor = robot_state_tensor.unsqueeze(0)
            human_states_tensor = human_states_tensor.unsqueeze(0)
            obstacle_states_tensor = obstacle_states_tensor.unsqueeze(0)
            wall_states_tensor = wall_states_tensor.unsqueeze(0)

        if device == torch.device('cuda:0'):
            robot_state_tensor = robot_state_tensor.cuda()
            human_states_tensor = human_states_tensor.cuda()
            obstacle_states_tensor = obstacle_states_tensor.cuda()
            wall_states_tensor = wall_states_tensor.cuda()
        elif device is not None:
            robot_state_tensor.to(device)
            human_states_tensor.to(device)
            obstacle_states_tensor.to(device)
            wall_states_tensor.to(device)

        if human_states_tensor is None:
            human_states_tensor = None

        return robot_state_tensor, human_states_tensor, obstacle_states_tensor, wall_states_tensor


class JointState_2tyeps(object):
    def __init__(self, robot_state, human_states):
        assert isinstance(robot_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.robot_state = robot_state
        self.human_states = human_states

    def to_tensor(self, add_batch_size=False, device=None):
        robot_state_tensor = torch.Tensor([self.robot_state.to_tuple()])
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in self.human_states])

        if add_batch_size:
            robot_state_tensor = robot_state_tensor.unsqueeze(0)
            human_states_tensor = human_states_tensor.unsqueeze(0)

        if device == torch.device('cuda:0'):
            robot_state_tensor = robot_state_tensor.cuda()
            human_states_tensor = human_states_tensor.cuda()
        elif device is not None:
            robot_state_tensor.to(device)
            human_states_tensor.to(device)

        if human_states_tensor.shape[1]==0:
            human_states_tensor = None
        return robot_state_tensor, human_states_tensor


def tensor_to_joint_state_2types(state):
    robot_state, human_states = state

    robot_state = robot_state.cpu().squeeze().data.numpy()
    robot_state = FullState(robot_state[0], robot_state[1], robot_state[2], robot_state[3], robot_state[4],
                            robot_state[5], robot_state[6], robot_state[7], robot_state[8])
    if human_states is None:
        human_states = []
    else:
        human_states = human_states.cpu().squeeze(0).data.numpy()
        human_states = [ObservableState(human_state[0], human_state[1], human_state[2], human_state[3],
                                        human_state[4]) for human_state in human_states]
    return JointState_2tyeps(robot_state, human_states)


def tensor_to_joint_state(state):
    robot_state, human_states, obstacle_states, wall_states = state

    robot_state = robot_state.cpu().squeeze().data.numpy()
    robot_state = FullState(robot_state[0], robot_state[1], robot_state[2], robot_state[3], robot_state[4],
                            robot_state[5], robot_state[6], robot_state[7], robot_state[8])
    if human_states is None:
        human_states = []
    else:
        human_states = human_states.cpu().squeeze(0).data.numpy()
        human_states = [ObservableState(human_state[0], human_state[1], human_state[2], human_state[3],
                                        human_state[4]) for human_state in human_states]

    if obstacle_states is None:
        human_states = []
    else:
        obstacle_states = obstacle_states.cpu().squeeze(0).data.numpy()
        obstacle_states = [ObstacleState(obstacle_state[0], obstacle_state[1], obstacle_state[2])
                           for obstacle_state in obstacle_states]

    if wall_states is None:
        wall_states = []
    else:
        wall_states = wall_states.cpu().squeeze(0).data.numpy()
        wall_states = [WallState(wall_state[0], wall_state[1], wall_state[2], wall_state[3])
                       for wall_state in wall_states]
    observed_state = (human_states, obstacle_states, wall_states)
    return JointState(robot_state, observed_state)
