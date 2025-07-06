import numpy as np
from math import sqrt, atan2, asin, sin, pi, cos, inf

def wraptopi(theta):
    if theta > np.pi:
        theta = theta - 2 * np.pi

    if theta < -np.pi:
        theta = theta + 2 * np.pi

    return theta
def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def config_vo_circle2(robot_state, circular):
    x, y, r = robot_state.px, robot_state.py, robot_state.radius
    mx, my, mr = circular.px, circular.py, circular.radius
    rel_x = x - mx
    rel_y = y - my
    dis_mr = np.sqrt((rel_y) ** 2 + (rel_x) ** 2)
    dis_r2g = np.sqrt((robot_state.px - robot_state.gx)**2 + (robot_state.py - robot_state.gy)**2)
    if dis_r2g < dis_mr - 0.1 or dis_mr > 4.0:
        return None, None
    angle_mr = atan2(my - y, mx - x)
    if dis_mr <= r + mr:
        dis_mr = r + mr

    ratio = (r + mr) / dis_mr
    half_angle = asin(ratio)
    line_left_ori = wraptopi(angle_mr + half_angle)
    line_right_ori = wraptopi(angle_mr - half_angle)
    return line_left_ori, line_right_ori

def config_vo_line2(robot_state, line_obstacle):
    x, y, r = robot_state.px, robot_state.py, robot_state.radius
    sx = line_obstacle.sx
    sy = line_obstacle.sy
    ex = line_obstacle.ex
    ey = line_obstacle.ey
    dis_r2g = np.sqrt((robot_state.px -robot_state.gx)**2 + (robot_state.py-robot_state.gy)**2)
    dis_r2line = point_to_segment_dist(sx, sy, ex, ey, robot_state.px, robot_state.py)
    if dis_r2g < dis_r2line or dis_r2line > 5.0:
        return None, None
    theta2 = atan2(sy - y, sx - x)
    theta1 = atan2(ey - y, ex - x)

    dis_mr1 = np.sqrt((sy - y) ** 2 + (sx - x) ** 2)
    dis_mr2 = np.sqrt((ey - y) ** 2 + (ex - x) ** 2)

    half_angle1 = asin(clamp(r / dis_mr1, 0, 1))
    half_angle2 = asin(clamp(r / dis_mr2, 0, 1))
    if True:
    # if wraptopi(theta2 - theta1) > 0:
        line_left_ori = wraptopi(theta2 + half_angle2)
        line_right_ori = wraptopi(theta1 - half_angle1)
    else:
         line_left_ori = wraptopi(theta1 + half_angle1)
         line_right_ori = wraptopi(theta2 - half_angle2)
    return line_left_ori, line_right_ori

def judge_in(ori, target_left_ori, target_right_ori):
    if wraptopi(ori-target_right_ori) < wraptopi(target_left_ori - target_right_ori) and wraptopi(target_left_ori - ori) < wraptopi(target_left_ori - target_right_ori):
    # if ori >= target_left_ori and ori <= target_right_ori:
        return True
    else:
        return False

def judge_in_regions(ori, left_ori_regions, right_ori_regions):
    for i in range(len(left_ori_regions)):
        if judge_in(ori, left_ori_regions[i], right_ori_regions[i]):
            return True
    return False

def subtarget(robot_state, human_states, obstacle_states, line_states):
    left_oris =[]
    right_oris = []
    for agent_state in human_states + obstacle_states:
        line_left_ori, line_right_ori = config_vo_circle2(robot_state, agent_state)
        if line_left_ori is None:
            continue
        # for i in range(len(left_oris)):
        #     # left_ori = left_oris[i]
        #     # right_ori = right_oris[i]
        #     # if judge_in(line_left_ori, left_ori, right_ori) or judge_in(line_right_ori, left_ori, right_ori):
        #     #     left_oris[i] = np.min([line_left_ori, left_ori])
        #     #     right_oris[i] = np.max([right_ori, line_right_ori])
        #     #     break
        #     # if i==len(left_oris)-1:
        #     if True:
        #         left_oris.append(line_left_ori)
        #         right_oris.append(line_right_ori)
        #     #     right_oris.append(line_right_ori)
        if True:
            left_oris.append(line_left_ori)
            right_oris.append(line_right_ori)
        if len(left_oris) == 0:
            left_oris.append(line_left_ori)
            right_oris.append(line_right_ori)
    for line_state in line_states:
        line_left_ori, line_right_ori = config_vo_line2(robot_state, line_state)
        if line_left_ori is None:
            continue
        # for i in range(len(left_oris)):
            # left_ori = left_oris[i]
            # right_ori = right_oris[i]
            # if judge_in(line_left_ori, left_ori, right_ori) or judge_in(line_right_ori, left_ori, right_ori):
            #     left_oris[i] = np.min([line_left_ori, left_ori])
            #     right_oris[i] = np.max([right_ori, line_right_ori])
            #     break
            # if i == len(left_oris)-1:
            #     left_oris.append(line_left_ori)
            #     right_oris.append(line_right_ori)
        if True:
            left_oris.append(line_left_ori)
            right_oris.append(line_right_ori)
        if len(left_oris) == 0:
            left_oris.append(line_left_ori)
            right_oris.append(line_right_ori)
    r2g_theta = atan2(robot_state.gy - robot_state.py, robot_state.gx - robot_state.px)
    k=36
    for i in range(k+1):
        left_ori = np.pi / k * i + r2g_theta
        if not judge_in_regions(left_ori, left_oris, right_oris):
            return left_ori
        right_ori = - np.pi / k * i + r2g_theta
        if not judge_in_regions(right_ori, left_oris, right_oris):
            return right_ori
    return r2g_theta
