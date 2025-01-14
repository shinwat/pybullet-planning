import os
import robomimic.utils.file_utils as FileUtils
import seaborn as sns
import matplotlib.pylab as plt
import math
import time
import numpy as np
from .pr2_utils import close_until_collision, set_joint_position, get_max_limit, \
    get_min_limit, joints_from_names
from .utils import PI, TRANSPARENT, approximate_as_prism, clone_body, euler_from_quat, wrap_angle, \
    get_angle, get_unit_vector, get_difference, get_joint_positions, Euler, Pose, get_link_pose, \
    get_link_subtree, get_pose, get_pose_distance, get_unit_vector, link_from_name, multiply, \
    point_from_pose, quat_from_pose, set_all_color, set_joint_positions, set_pose, unit_pose, \
    invert, pose_from_base_values, sample_reachable_base, unit_from_theta

TIAGO_URDF = "models/tiago_description/tiago.urdf"

EXTENDED_ARM = [0., 0., 0., 0., 0., 0., 0.]
TUCKED_ARM = [1.5, 1., 0.06, 1.0, -1.7, 0., 0.] # TODO: get actual values

TIAGO_GROUPS = {
    'base': ['x', 'y', 'theta'],
    'torso': ['torso_lift_joint'],
    'head': ['head_1_joint', 'head_2_joint'], # pan, tilt
    'arm': ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', # shoulder x3
            'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'], # elbow, wrist x3
    'gripper': ['gripper_right_finger_joint', 'gripper_left_finger_joint'],
}

EPSILON = 1e-6
GRASP_LENGTH = 0.
GRIPPER_MARGIN = 0.1
MAX_GRASP_WIDTH = np.inf
TOOL_POSE = Pose(euler=Euler(pitch=PI))
#TODO: find actual values
TOP_HOLDING_LEFT_ARM = [0.8, 0.74313199, 1.5, -1.46688405, 0.94223229, 1.75442826, 2.22254125]
SIDE_HOLDING_LEFT_ARM = [0.39277395, 0.33330058, 0., -1.52238431, 2.72170996, -1.21946936, -2.98914779]

TIAGO_LEFT_CARRY_CONFS = {
    'top': TUCKED_ARM, #TODO: use TOP_HOLDING_LEFT_ARM,
    'side': SIDE_HOLDING_LEFT_ARM,
}

TIAGO_TOOL_FRAME = 'gripper_tool_frame'
TIAGO_GRIPPER_ROOT = 'gripper_link'
TIAGO_BASE_LINK = 'base_footprint'

def get_base_pose(robot):
    return get_link_pose(robot, link_from_name(robot, TIAGO_BASE_LINK))

def open_gripper(robot):
    for joint in joints_from_names(robot, TIAGO_GROUPS['gripper']):
        set_joint_position(robot, joint, get_max_limit(robot, joint))

def close_gripper(robot):
    for joint in joints_from_names(robot, TIAGO_GROUPS['arm']):
        set_joint_position(robot, joint, get_min_limit(robot, joint))

def get_group_joints(robot, group):
    return joints_from_names(robot, TIAGO_GROUPS[group])

def get_gripper_link(robot):
    return link_from_name(robot, TIAGO_TOOL_FRAME)

def get_group_conf(robot, group):
    return get_joint_positions(robot, get_group_joints(robot, group))

def set_group_conf(robot, group, positions):
    set_joint_positions(robot, get_group_joints(robot, group), positions)

def get_arm_joints(robot):
    return get_group_joints(robot, 'arm')

def get_torso_arm_joints(robot):
    return joints_from_names(robot, TIAGO_GROUPS['torso'] + TIAGO_GROUPS['arm'])

def get_gripper_joints(robot):
    return get_group_joints(robot, 'gripper')

def get_carry_conf(grasp_type):
    return TIAGO_LEFT_CARRY_CONFS[grasp_type]

def get_arm_conf(robot):
    return get_joint_positions(robot, get_arm_joints(robot))

def set_arm_conf(robot, conf):
    set_joint_positions(robot, get_arm_joints(robot), conf)

def get_top_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                   max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH):
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    reflect_z = Pose(euler=[0, math.pi, 0])
    translate_z = Pose(point=[0, 0, h / 2 - grasp_length])
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    grasps = []
    if w <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, math.pi / 2 + i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]
    if l <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]
    return grasps

def get_align(body, target_pose, num_samples=5, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                   grasp_length=GRASP_LENGTH, safety_margin=GRIPPER_MARGIN):
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    side = max(w, l)
    reflect_z = Pose(euler=[0, math.pi, 0])
    translate_z = Pose(point=[0, 0, h / 2 - grasp_length]) # lift gripper by grasp length

    current_pose = get_pose(body)
    angle = get_angle(point_from_pose(target_pose.value), point_from_pose(current_pose))
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    grasp = multiply(tool_pose, translate_z,
                        reflect_z, translate_center, body_pose)
    translate_offset = np.array([(side/2+safety_margin)*math.cos(angle), (side/2+safety_margin)*math.sin(angle), 0])
    rotate_z = np.array([0, 0, wrap_angle(angle)])
    align = Pose(point=translate_offset, euler=rotate_z)
    return align

def compute_grasp_width(robot, body, grasp_pose, **kwargs):
    tool_link = get_gripper_link(robot)
    tool_pose = get_link_pose(robot, tool_link)
    body_pose = multiply(tool_pose, grasp_pose)
    set_pose(body, body_pose)
    gripper_joints = get_gripper_joints(robot)
    return close_until_collision(robot, gripper_joints, bodies=[body], **kwargs)

def create_gripper(robot, visual=True):
    link_name = TIAGO_GRIPPER_ROOT
    links = get_link_subtree(robot, link_from_name(robot, link_name))
    gripper = clone_body(robot, links=links, visual=False, collision=False)  # TODO: collision to True
    if not visual:
        set_all_color(robot, TRANSPARENT)
    return gripper

def get_midpoint_pose(start_pose, end_pose):
    difference = get_difference(point_from_pose(start_pose), point_from_pose(end_pose))
    # print("Diff: ",difference)
    return Pose(point=difference/2+point_from_pose(start_pose),
                euler=euler_from_quat(quat_from_pose(start_pose))
                )

def align_gripper(pose, orientation):
    return Pose(point=point_from_pose(pose),
                euler=euler_from_quat(quat_from_pose(orientation))
                )

# samples base pose from learned value function
def learned_pose_generator(robot, start_pose, gripper_pose, goal_pose, policy_dir, reach_dir, grid_search):
    # pseudocode
    # input: goal block position (3) = goal_pose,
    #        initial block pose (7) = start_pose, 
    #        initial end-effector pose (7) = gripper_pose
    # output: base pose (x, y, theta)
    while True:
        distance = 1.2
        x = (-distance, distance)
        y = x
        theta = (-np.pi, np.pi)
        radius = (0.25, 1.0)
        phi = (-math.pi, math.pi)
        r_min = radius[0] - EPSILON
        r_max = radius[-1] + EPSILON
        if reach_dir is not None: # for reachability test
            # compare with baseline
            base_values = sample_reachable_base(robot, point_from_pose(gripper_pose))
            goal_to_baseline = np.array(base_values) - np.array(goal_pose[0])
            rb = np.linalg.norm(goal_to_baseline[:-1])
            if rb > r_min and rb < r_max:
                print('base: goal block IS reachable: ', rb)
            else:
                print('base: goal block is NOT reachable: ', rb)
            # save to file
            results = {}
            if not os.path.exists(reach_dir):
                print("Making new directory at {}".format(reach_dir))
                os.makedirs(reach_dir)
            results['uniform'] = int(rb > r_min and rb < r_max)
        for root, _, files in os.walk(policy_dir):
            for file in files:
                if file.endswith('.pth'):
                    policy_path = os.path.join(root, file)
                    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=policy_path)

                    if grid_search: # OPTION 1: GRID SEARCH
                        param1_range = np.linspace(start=x[0], stop=x[-1], num=20)
                        param2_range = np.linspace(start=y[0], stop=y[-1], num=20)
                        param3_range = np.linspace(start=theta[0], stop=theta[-1], num=20)

                        def get_metric(params):
                            feasibility = policy.get_value(
                                ob=get_state(params, start_pose, gripper_pose, True), 
                                goal=get_goal(params, ((0.0, 0.0, goal_pose[0][-1]), goal_pose[-1]), gripper_pose, True)
                            ) # KLUDGE: pretend no access to goal
                            return feasibility[0]

                        best_metric = 0
                        best_params = None
                        grid_data = np.zeros((param1_range.size, param2_range.size, param3_range.size))
                        start = time.time()

                        for i, param1 in enumerate(param1_range):
                            for j, param2 in enumerate(param2_range):
                                for k, param3 in enumerate(param3_range):
                                    params = (param1, param2, param3)
                                    metric = get_metric(params)
                                    grid_data[i, j, k] = metric
                        end = time.time()
                        print('execution time (s): ', end-start)
                        grid = np.max(grid_data, 2) # only x and y
                        # loop through the 2D grid
                        best_temp_metric = float('-Inf')
                        best_temp_params = None
                        best_index = None
                        for i in range(np.shape(grid)[0]):
                            for j in range(np.shape(grid)[-1]):
                                if grid[i,j] > best_temp_metric:
                                    best_temp_metric = grid[i,j]
                                    best_temp_params = (param1_range[i], param2_range[j])
                                    best_index = (i,j)
                        
                        # check that the index is in the reachable zone
                        block_to_base = np.array(best_temp_params) - np.array(start_pose[0][:1])
                        print('start pose: ', start_pose[0][:2])
                        r = np.linalg.norm(block_to_base)
                        if r > 0.25 and r < 1.0:
                            print('parameter is reachable: ', r)
                        else:
                            print('parameter is NOT reachable: ', r)

                        # search for theta in original grid
                        for k in range(np.shape(grid_data)[-1]):
                            params = best_temp_params + (param3_range[k],)
                            if abs(grid_data[best_index[0],best_index[-1],k]) == best_temp_metric:
                                best_metric = grid_data[best_index[0],best_index[-1],k]
                                best_params = params
                                break

                        # Output the best metric and the corresponding parameters
                        print('Best metric:', best_metric)
                        print('Best parameters:', best_params)

                        ax = sns.heatmap(grid.T, linewidth=0.5)
                        ax.invert_yaxis()
                        plt.show()
                        for _ in range(25):
                            # sample from the grid with more probability for higher values
                            perturbed_params = perturb_base(robot, best_params, reachable_range=(0.0, 0.25))
                            yield perturbed_params
                        else:
                            break
                    
                    else: # OPTION 2: SMARTER OPTIMIZATION
                        from scipy.optimize import minimize
                        theta = (-np.pi, np.pi)
                        radius = (0.25, 1.0)
                        phi = (-math.pi, math.pi)
                        bounds = (radius, phi, theta)
                        method = 'Nelder-Mead' #'Nelder-Mead' #'L-BFGS-B' #'Powell'
                        initial_guess = [np.mean(bound) for bound in bounds]
                        objective = lambda params, policy, start_pose, gripper_pose, goal_pose: -policy.get_value(
                            ob=get_state(params, start_pose, gripper_pose), 
                            goal=get_goal(params, ((0.0, 0.0, goal_pose[0][-1]), goal_pose[-1]), gripper_pose)
                        )
                        from functools import partial
                        partial_obj = partial(
                            objective, 
                            policy=policy,
                            start_pose=start_pose,
                            gripper_pose=gripper_pose,
                            goal_pose=goal_pose
                        )
                        result = minimize(
                            partial_obj,
                            initial_guess,
                            method=method,
                            bounds=bounds,
                            options={'xatol': 1e-6, 'disp': True}
                        ) #'ftol':0.001, 
                        print('sampled pose (r, p, t): ', result.x)
                        # convert back to world pose
                        (radius, phi, theta) = result.x
                        x, y = radius*unit_from_theta(theta) + gripper_pose[0][:2]
                        learned_base_values = (x, y, theta)
                        print('sampled base values: ', learned_base_values)
                        # check that the index is in the reachable zone
                        print('initial block pose: ', start_pose[0])
                        goal_to_base = np.array(learned_base_values) - np.array(goal_pose[0])
                        print('goal block pose: ', goal_pose[0])
                        # check w.r.t. initial block
                        block_to_base = np.array(learned_base_values) - np.array(start_pose[0])
                        r = np.linalg.norm(block_to_base[:-1])
                        if r > r_min and r < r_max:
                            print('initial block is reachable: ', r)
                        else:
                            print('initial block is NOT reachable: ', r)
                        # check w.r.t. goal block
                        rg = np.linalg.norm(goal_to_base[:-1])
                        
                        if rg > r_min and rg < r_max:
                            print('goal block is reachable: ', rg)
                        else:
                            print('goal block is NOT reachable: ', rg)
                        if reach_dir is not None: # for reachability test
                            results[policy_path] = int(rg > r_min and rg < r_max)
        if reach_dir is not None: # for reachability test
            t1, t2 = str(time.time()).split(".")
            col_path = os.path.join(reach_dir, "reachable_{}_{}.npz".format(t1, t2))
            np.savez(
                col_path,
                results=results
            )
            time.sleep(0.1)
            yield None
        else:
            if rg > r_min and rg < r_max:
                yield learned_base_values
                while True:
                    yield perturb_base(robot, learned_base_values, reachable_range=(0.0, 0.25))
            else: # revert to uniform
                while True:
                    base_values = sample_reachable_base(robot, point_from_pose(gripper_pose))
                    if base_values is None:
                        break
                    yield base_values

def get_theta(params, goal_pose):
    theta = math.atan2(goal_pose[0][1] - params[-1], goal_pose[0][0] - params[0])
    return theta

def get_state(params, start_pose, gripper_pose, grid=False):
    if grid:
        base_values = params
    else:
        (radius, theta, phi) = params
        x, y = radius*unit_from_theta(theta) + gripper_pose[0][:2]
        base_values = (x, y, phi)
    world_from_base = pose_from_base_values(base_values)
    tool_pos, tool_orn = multiply(invert(world_from_base), gripper_pose)
    tool_orn = euler_from_quat(tool_orn)
    obj_pos, obj_orn = multiply(invert(world_from_base), start_pose)
    obj_orn = euler_from_quat(obj_orn)
    rel_pos = tuple(map(lambda x, y: x-y, obj_pos, tool_pos))
    obs = {}
    obs["tool_pos"] = np.array(tool_pos) # 3
    obs["tool_orn"] = np.array(tool_orn) # 4
    obs["obj_pos"] = np.array(obj_pos) # 3
    obs["obj_orn"] = np.array(obj_orn) # 4
    obs["rel_pos"] = np.array(rel_pos) #3
    return obs

def get_goal(params, goal_pose, gripper_pose, grid=False):
    if grid:
        base_values = params
    else:
        (radius, theta, phi) = params
        x, y = radius*unit_from_theta(theta) + gripper_pose[0][:2]
        base_values = (x, y, phi)
    world_from_base = pose_from_base_values(base_values)
    goal_pos, _ = multiply(invert(world_from_base), goal_pose)
    return  {"obj_pos" : np.array(goal_pos)}

def perturb_base(robot, point, reachable_range=(0.0, 0.2)):
    #TODO: figure out how much is acceptable amount
    radius = np.random.uniform(*reachable_range)
    x, y = radius*unit_from_theta(np.random.uniform(-np.pi, np.pi)) + point[:2]
    yaw = np.random.uniform(-np.pi/2, np.pi/2) + point[-1]
    base_values = (x, y, yaw)
    return base_values