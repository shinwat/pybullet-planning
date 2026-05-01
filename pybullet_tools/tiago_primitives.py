from __future__ import annotations
from collections import OrderedDict
import copy
import math
import random
import time
import os
from itertools import islice, count
from operator import add, sub, truediv, mul
from statistics import mean

import numpy as np

from pddlstream.algorithms.skills import recover_skill_model_from_stream_pairs, TEMP_SKILLS_DIR

from .ikfast.utils import USE_CURRENT
from .ikfast.tiago.ik import get_base, get_pose_wrt_base, get_tool_pose, get_tool_pose_wrt_base, \
    is_ik_compiled, tiago_inverse_kinematics, sample_tool_ik
from .tiago_utils import TIAGO_GRIPPER_ROOT, TIAGO_GROUPS, TIAGO_TOOL_FRAME, \
    align_gripper, close_gripper, compute_grasp_width, get_align, \
    get_arm_joints, get_carry_conf, get_gripper_consts, get_gripper_joints, get_gripper_link, get_group_conf, \
    get_group_joints, get_top_grasps, is_reachable, maybe_flip_phi, open_gripper, perturb_base, \
    show_heatmap, sort_3d_array_indices_desc, pose2d_from_pose, get_gripper_state
from .utils import BASE_LINK, UNIT_LIMITS, Attachment, BodySaver, Euler, Point, WorldSaver, add_segments, \
    custom_pose_generator, euler_from_quat, add_fixed_constraint, all_between, \
    base_values_from_pose, create_attachment, disable_real_time, \
    enable_gravity, flatten_links, get_aabb, get_bodies, get_body_name, get_bounding_box, \
    get_configuration, get_custom_limits, get_distance, get_extend_fn, \
    get_joint_limits, get_joint_position, get_joint_positions, get_link_pose, get_link_subtree, get_min_limit, \
    get_moving_links, get_name, get_pose, get_relative_pose, get_static_image, \
    get_time_step, get_unit_vector, has_gui, interpolate_poses, invert, \
    is_placement, joint_controller_hold, joints_from_names, \
    link_from_name, multiply, pairwise_collision, plan_direct_joint_motion, \
    plan_joint_motion, point_from_pose, point_in_annulus, remove_debug, \
    remove_fixed_constraint, sample_placement, sample_reachable_base, sample_reachable_base2, \
    set_pose, step_simulation, sub_inverse_kinematics, uniform_pose_generator, unit_from_theta, \
    unit_quat, wait_for_duration, wait_if_gui, set_joint_positions, waypoints_from_path
from .utils import Pose as Posee
BASE_EXTENT = 3.5 # 2.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False
CONTROL_FREQ = 20.0 # Hz
MAX_HORIZON = 50
EVAL_HORIZON = 100
PLATE_VERTICES = [[-0.135, -0.135],[0.135,-0.135],[0.135,0.135],[-0.135,0.135]]
PLATE_RADIUS = 0.07
HOOK_WIDTH = 0.1
HOOK_LENGTH = 0.2
MAX_JOINT_VELOCITIES = np.array([1.95, 1.95, 2.35, 2.35, 1.95, 1.95, 1.76]) # in rad/s
ACTION_NORM_CONST = (MAX_JOINT_VELOCITIES / CONTROL_FREQ).tolist()
GRIPPER_NORM_COST = [0.05, 0.05, 0.05, 0.05] # [m, m, rad, rad]/step
BLOCK_REACHABLE_RANGE = (0.48, 0.99)
GRIPPER_REACHABLE_RANGE = (0.45, 0.85) #85 | 70
MAX_PERTURBATION = 0.15
HEURISTIC_ATTEMPTS = 50

##################################################

class State(object):
    def __init__(self, attachments={}, cleaned=set(), cooked=set()):
        self.poses = {body: Pose(body, get_pose(body))
                      for body in get_bodies() if body not in attachments}
        self.grasps = {}
        self.attachments = attachments
        self.cleaned = cleaned
        self.cooked = cooked
    def assign(self):
        for attachment in self.attachments.values():
            #attach.attachment.assign()
            attachment.assign()

class Pose(object):
    num = count()
    def __init__(self, body, value=None, support=None, init=False):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.support = support
        self.init = init
        self.index = next(self.num)
    @property
    def bodies(self):
        return flatten_links(self.body)
    def assign(self):
        set_pose(self.body, self.value)
    def iterate(self):
        yield self
    def to_base_conf(self):
        values = base_values_from_pose(self.value)
        return Conf(self.body, range(len(values)), values)
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'p{}'.format(index)

# like attachment but doesn't set the object pose
class Adjunct(object):
    def __init__(self, parent, parent_link, grasp_pose, child):
        self.parent = parent # TODO: support no parent
        self.parent_link = parent_link
        self.grasp_pose = grasp_pose
        self.child = child
    @property
    def bodies(self):
        return flatten_links(self.child) | flatten_links(self.parent, get_link_subtree(
            self.parent, self.parent_link))
    def assign(self):
        return get_pose(self.child)
    def apply_mapping(self, mapping):
        self.parent = mapping.get(self.parent, self.parent)
        self.child = mapping.get(self.child, self.child)
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.parent, self.child)

class Alignment(object):
    def __init__(self, grasp_type, body, value, approach, carry):
        self.grasp_type = grasp_type
        self.body = body
        self.value = tuple(value) # gripper_from_object
        self.approach = tuple(approach)
        self.carry = tuple(carry)
    def get_attachment(self, robot):
        tool_link = link_from_name(robot, TIAGO_TOOL_FRAME)
        return Adjunct(robot, tool_link, self.value, self.body)
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

class Grasp(object):
    def __init__(self, grasp_type, body, value, approach, carry):
        self.grasp_type = grasp_type
        self.body = body
        self.value = tuple(value) # gripper_from_object
        self.approach = tuple(approach)
        self.carry = tuple(carry)
    def get_attachment(self, robot):
        tool_link = link_from_name(robot, TIAGO_TOOL_FRAME)
        return Attachment(robot, tool_link, self.value, self.body)
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)
    
class Conf(object):
    def __init__(self, body, joints, values=None, init=False):
        self.body = body
        self.joints = joints
        if values is None:
            values = get_joint_positions(self.body, self.joints)
        self.values = tuple(values)
        self.init = init
    @property
    def bodies(self): # TODO: misnomer
        return flatten_links(self.body, get_moving_links(self.body, self.joints))
    def assign(self):
        set_joint_positions(self.body, self.joints, self.values)
    def iterate(self):
        yield self
    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)
    
##################################################

class Command(object):
    def control(self, dt=0):
        raise NotImplementedError()
    def apply(self, state, **kwargs):
        raise NotImplementedError()
    def iterate(self):
        raise NotImplementedError()

class Commands(object):
    def __init__(self, state, savers=[], commands=[]):
        self.state = state
        self.savers = tuple(savers)
        self.commands = tuple(commands)
    def assign(self):
        for saver in self.savers:
            saver.restore()
        return copy.copy(self.state)
    def apply(self, state, **kwargs):
        for command in self.commands:
            for result in command.apply(state, **kwargs):
                yield result
    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)

##################################################

class GripperCommand(Command):
    def __init__(self, robot, position, teleport=False):
        self.robot = robot
        self.position = position
        self.teleport = teleport
    def apply(self, state, **kwargs):
        joints = get_gripper_joints(self.robot)
        start_conf = get_joint_positions(self.robot, joints)
        end_conf = [self.position] * len(joints)
        if self.teleport:
            path = [start_conf, end_conf]
        else:
            extend_fn = get_extend_fn(self.robot, joints)
            path = [start_conf] + list(extend_fn(start_conf, end_conf))
        for positions in path:
            set_joint_positions(self.robot, joints, positions)
            yield positions
    def control(self, **kwargs):
        joints = get_gripper_joints(self.robot)
        positions = [self.position]*len(joints)
        for _ in joint_controller_hold(self.robot, joints, positions):
            step_simulation()

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, get_body_name(self.robot), self.position)
    
class Attach(Command):
    vacuum = True
    def __init__(self, robot, grasp, body):
        self.robot = robot
        self.grasp = grasp
        self.body = body
        self.link = link_from_name(self.robot, TIAGO_TOOL_FRAME)
        #self.attachment = None
    def assign(self):
        gripper_pose = get_link_pose(self.robot, self.link)
        body_pose = multiply(gripper_pose, self.grasp.value)
        set_pose(self.body, body_pose)
    def apply(self, state, **kwargs):
        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        state.grasps[self.body] = self.grasp
        del state.poses[self.body]
        yield
    def control(self, dt=0, **kwargs):
        if self.vacuum:
            add_fixed_constraint(self.body, self.robot, self.link)
        else:
            # TODO: the gripper doesn't quite work yet
            joints = joints_from_names(self.robot, TIAGO_GROUPS['gripper'])
            values = [get_min_limit(self.robot, joint) for joint in joints] # Closed
            for _ in joint_controller_hold(self.robot, joints, values):
                step_simulation()
                time.sleep(dt)
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, get_body_name(self.robot), get_name(self.body))

class Detach(Command):
    def __init__(self, robot, body):
        self.robot = robot
        self.body = body
        self.link = link_from_name(self.robot, TIAGO_TOOL_FRAME)
        # TODO: pose argument to maintain same object
    def apply(self, state, **kwargs):
        del state.attachments[self.body]
        state.poses[self.body] = Pose(self.body, get_pose(self.body))
        del state.grasps[self.body]
        yield
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, get_body_name(self.robot), get_name(self.body))
    
class Trajectory(Command):
    _draw = False
    def __init__(self, path):
        self.path = tuple(path)
        # TODO: constructor that takes in this info
    def apply(self, state, sample=1):
        handles = add_segments(self.to_points()) if self._draw and has_gui() else []
        for conf in self.path[::sample]:
            conf.assign()
            yield
        end_conf = self.path[-1]
        if isinstance(end_conf, Pose):
            state.poses[end_conf.body] = end_conf
        for handle in handles:
            remove_debug(handle)
    def control(self, dt=0, **kwargs):
        # TODO: just waypoints
        for conf in self.path:
            if isinstance(conf, Pose):
                conf = conf.to_base_conf()
            for _ in joint_controller_hold(conf.body, conf.joints, conf.values):
                step_simulation()
                time.sleep(dt)
    def to_points(self, link=BASE_LINK):
        # TODO: this is computationally expensive
        points = []
        for conf in self.path:
            with BodySaver(conf.body):
                conf.assign()
                #point = np.array(point_from_pose(get_link_pose(conf.body, link)))
                point = np.array(get_group_conf(conf.body, 'base'))
                point[2] = 0
                point += 1e-2*np.array([0, 0, 1])
                if not (points and np.allclose(points[-1], point, atol=1e-3, rtol=0)):
                    points.append(point)
        points = get_target_path(self)
        return waypoints_from_path(points)
    def distance(self, distance_fn=get_distance):
        total = 0.
        for q1, q2 in zip(self.path, self.path[1:]):
            total += distance_fn(q1.values, q2.values)
        return total
    def iterate(self):
        for conf in self.path:
            yield conf
    def reverse(self):
        return Trajectory(reversed(self.path))
    def __repr__(self):
        d = 0
        if self.path:
            conf = self.path[0]
            d = 3 if isinstance(conf, Pose) else len(conf.joints)
        return 't({},{})'.format(d, len(self.path))
    
def create_trajectory(robot, joints, path):
    return Trajectory(Conf(robot, joints, q) for q in path)

def get_target_path(trajectory):
    return [get_target_point(conf) for conf in trajectory.path]

def get_target_point(conf):
    robot = conf.body
    link = link_from_name(robot, 'torso_lift_link')
    with BodySaver(conf.body):
        conf.assign()
        lower, upper = get_aabb(robot, link)
        center = np.average([lower, upper], axis=0)
        point = np.array(get_group_conf(conf.body, 'base'))
        point[2] = center[2]
        return point

##################################################

#TODO: move to utils
def normalize_joint(joint, lower, upper):
    tmp = (joint - lower) / (upper - lower)
    new_lower, new_upper = UNIT_LIMITS
    return (tmp - 0.5)*(new_upper - new_lower)

#TODO: move to utils
def unnormalize_joint(joint, lower, upper):
    new_lower, new_upper = UNIT_LIMITS
    tmp = joint/(new_upper - new_lower) + 0.5
    return tmp*(upper - lower) + lower

#TODO: move to utils
def get_normalized_arm_joint_positions(body):
    return tuple(
            normalize_joint(
                get_joint_position(body, joint), 
                *get_joint_limits(body, joint)) for joint in  get_arm_joints(body)
        )

#TODO: move to utils
def get_unnormalized_arm_joint_positions(body, positions):
    joints = get_arm_joints(body)
    return tuple(
            unnormalize_joint(
                positions[idx], 
                *get_joint_limits(body, joints[idx])) for idx in range(len(joints))
        )

#TODO: move to utils
def is_point_in_plate(point):
    return int(np.linalg.norm(point[:2]) < PLATE_RADIUS)

def get_state_wrt_base(robot, body):
    tool_pose = pose2d_from_pose(get_tool_pose_wrt_base(robot))
    obj_pose = pose2d_from_pose(get_pose_wrt_base(robot, get_pose(body)))
    ret = {}
    ret["tool_pose"] = np.array(tool_pose) # 4
    ret["obj_pose"] = np.array(obj_pose) # 4
    return ret

def get_goal_wrt_base(robot, pose):
    goal_pos, _ = get_pose_wrt_base(robot, pose)
    goal_pos = np.array(goal_pos)
    return goal_pos[:-1]

def get_goal_wrt_world(pose):
    goal_pos, _ = pose
    goal_pos = np.array(goal_pos)
    return {"obj_pos" : goal_pos[:-1]}

def augment_state(obs, robot):
    image = get_static_image(get_tool_pose(robot))[:,:,:3] 
    image = np.moveaxis(image, -1, 0) #KLUDGE: for some reason, VisualCore takes channel-first as input?
    obs["wrist_image"] = image
    return obs

def flatten_state(state):
    flat = []
    for feature in state:
        flat.append(state[feature])
    return np.concatenate(flat)

def condition_state(state, goal):
    state["goal"] = goal
    return state

def normalize_state(state, stats):
    return np.clip(state, stats[0], stats[1])

def get_goal(robot, pose):
    goal_pos, _ = get_pose_wrt_base(robot, pose) # wrt base frame
    goal_pos = np.array(goal_pos)
    goal_pos[-1] -= 0.001
    return {"obj_pos" : goal_pos}

def convert_action_to_desired_joint_state(action, robot, gripper_consts, old_gripper_state):
    (roll, pitch, z) = gripper_consts
    unnormed_action = list(map(mul, action, GRIPPER_NORM_COST))
    (x, y, sin_yaw, cos_yaw) = list(map(add, unnormed_action, old_gripper_state)) # x,y,rot
    yaw = np.arctan2(sin_yaw, cos_yaw)
    new_gripper_pose = Posee(
        Point(x, y, z),
        Euler(roll, pitch, yaw)
    )
    gripper_pose_wrt_world = multiply(get_base(robot), new_gripper_pose)
    return sample_tool_ik(robot, gripper_pose_wrt_world, max_attempts=100, torso_limits=USE_CURRENT) # upper_limits

class Push(Command):
    def __init__(self, robot, body, pose, trajectory, directory=None, model=None, evaluate=False, bootstrap=False, ablation=False, buffer=None, stats=None, demo_dict=None, dense=False, jammed=False):
        self.robot = robot
        self.body = body
        self.pose = pose
        self.trajectory = trajectory
        self.directory = directory
        self.model = model
        self.buffer = buffer
        self.evaluate = evaluate
        self.bootstrap = bootstrap
        self.ablation = ablation
        self.stats = stats
        self.demo_dict = demo_dict
        self.dense = dense
        self.jammed = jammed
    def apply(self, state, **kwargs):
        self.trajectory.apply(state, **kwargs)
    def control(self, **kwargs):
        saver = WorldSaver()
        sim_dt = get_time_step()
        sim_time = 0.0
        log_dict = None
        while True:
            obj_pose = get_pose_wrt_base(self.robot, get_pose(self.body))
            if obj_pose[0][-1] < 0.1:
                print('block is on the floor.')
                break
            goal = get_goal_wrt_base(self.robot, self.pose.value)
            horizon = EVAL_HORIZON if self.evaluate and self.directory is None else MAX_HORIZON
            joints = get_arm_joints(self.robot)
            state = flatten_state(condition_state(get_state_wrt_base(self.robot, self.body), goal))
            if self.stats is not None:
                state = normalize_state(state, self.stats)
            init_state = state
            torso_state = get_group_conf(self.robot, 'torso')
            arm_state = get_group_conf(self.robot, 'arm')
            gripper_state = get_group_conf(self.robot, 'gripper')
            base_state = get_group_conf(self.robot, 'base')
            obj_state = get_pose(self.body) # wrt world frame
            goal_state = self.pose.value
            config = np.concatenate((torso_state, arm_state, gripper_state, obj_state[0], obj_state[1], base_state, goal_state[0]))

            if self.model is not None:
                def step(desired_joint_state, old_joint_value):
                    if desired_joint_state is None:
                        state =  condition_state(get_state_wrt_base(self.robot, self.body), goal)
                        return state, -1, True, 0.
                    if self.jammed:
                        if desired_joint_state[0] < old_joint_value and old_joint_value > np.pi/2: # only left pushes
                            desired_joint_state[0] = old_joint_value
                    old_joint_value = desired_joint_state[0]
                    sim_time = 0.0
                    for _ in joint_controller_hold(self.robot, joints, desired_joint_state, velocity_scale=0.1):
                        step_simulation()
                        sim_time += sim_dt
                        if sim_time > 1/CONTROL_FREQ:
                            break
                    state = get_state_wrt_base(self.robot, self.body)
                    state = condition_state(state, goal)
                    dist = np.linalg.norm(state['obj_pose'][:2] - state['goal']) # tool_pose for reach
                    done = dist < PLATE_RADIUS
                    if self.dense:
                        reward = - dist
                    elif done:
                        reward = 0
                    else:
                        reward = -1
                    return state, reward, done, old_joint_value

                def rollout_corl(model, buffer=None):
                    logs = []
                    states = []
                    actions = []
                    next_states = []
                    rewards = []
                    dones = []
                    state = flatten_state(condition_state(get_state_wrt_base(self.robot, self.body), goal))
                    if self.stats is not None:
                        state = normalize_state(state, self.stats)
                    old_gripper_pose = get_tool_pose_wrt_base(self.robot)
                    old_gripper_state = get_gripper_state(old_gripper_pose)
                    gripper_consts = get_gripper_consts(old_gripper_pose)
                    block_y = state[5]
                    gripper_y = old_gripper_pose[0][1]
                    jammed_init = gripper_y > 0 and gripper_y > block_y # if y-axis is negative, jammed
                    jammed_init = jammed_init and self.jammed
                    if jammed_init:
                        print('arm is jammed.')
                    old_joint_value = get_joint_positions(self.robot, joints)[0]

                    if self.evaluate:
                        model.actor.eval()
                    for i in range(horizon):
                        fallback = False
                        for _ in range(10):
                            if self.evaluate:
                                action = model.actor.act(state, device=model.device, fallback=fallback)
                            else:
                                action = model.actor.sample(state, device=model.device)
                            desired_joint_state = convert_action_to_desired_joint_state(
                                action, 
                                self.robot, 
                                gripper_consts, 
                                old_gripper_state
                            )
                            if desired_joint_state is None:
                                fallback = True
                            else:
                                break

                        next_state, reward, done, old_joint_value = step(desired_joint_state, old_joint_value)
                        
                        next_state = flatten_state(next_state)
                        if self.stats is not None:
                            next_state = normalize_state(next_state, self.stats)

                        if buffer is not None and buffer.__class__.__name__ in ("ReplayBuffer", "PrioritizedReplayBuffer"):
                            buffer.add_transition(state, action, reward, next_state, done)
                        
                        old_gripper_state = get_gripper_state(get_tool_pose_wrt_base(self.robot))
                        
                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        next_states.append(next_state)
                        dones.append(done)

                        state = next_state

                        if not self.evaluate:
                            log_dict = model.train(buffer)
                            logs.append([log_dict, model.total_it])

                        # if done or success, end before horizon is reached
                        if done:
                            if desired_joint_state is None:
                                print("IK failed: ", i)
                                wait_if_gui()
                            break
                    wait_if_gui()
                    if self.evaluate:
                        model.actor.train()
                    success = is_point_in_plate(get_pose(self.body)[0]) # get_tool_pose(self.robot)[0] for reach

                    # per-episode buffer
                    if buffer is not None and buffer.__class__.__name__ == "HindsightReplayBuffer":
                        # prepare batch for HER buffer
                        padded_states = pad_list_to_length(states, MAX_HORIZON, next_states[-1])
                        padded_next_states = pad_list_to_length(next_states, MAX_HORIZON)
                        padded_actions = pad_list_to_length_with_zeros(actions, MAX_HORIZON)
                        padded_rewards = pad_list_to_length(rewards, MAX_HORIZON)
                        padded_dones = pad_list_to_length(dones, MAX_HORIZON)

                        trajectory = {
                            'states': [padded_states],
                            'next_states': [padded_next_states],
                            'actions': [padded_actions],
                            'rewards': [padded_rewards],
                            'dones': [padded_dones],
                        }
                        buffer.store_episode(list_to_numpy(trajectory))

                    return success, jammed_init, logs
                
                wait_if_gui()
                success, jammed_init, log_dict = rollout_corl(self.model, self.buffer)
                print('success:', success)

            if not self.bootstrap:
                states = []
                next_states = []
                rewards = []
                dones = []
                obj_poses = []
                action_infos = []
                sim_time = 0.0
                state = get_state_wrt_base(self.robot, self.body)
                old_gripper_pose = get_tool_pose_wrt_base(self.robot)
                old_gripper_state = get_gripper_state(old_gripper_pose)
                steps = 0
                wait_if_gui()
                # demo replay
                if self.demo_dict is not None:
                    actions = self.demo_dict["actions"]
                    gripper_consts = get_gripper_consts(old_gripper_pose)
                    for action in actions:
                        desired_joint_state = convert_action_to_desired_joint_state(action, self.robot, gripper_consts, old_gripper_state)
                        if desired_joint_state is None:
                            print("IK failed.")
                            break
                        for i, _ in enumerate(joint_controller_hold(self.robot, joints, desired_joint_state, velocity_scale=0.1)):
                            step_simulation()
                            sim_time += sim_dt
                            if sim_time > 1/CONTROL_FREQ:
                                sim_time = 0.0
                                steps += 1
                                break # controller is reset every control loop
                        old_gripper_state = get_gripper_state(get_tool_pose_wrt_base(self.robot))
                    wait_if_gui()
                    return is_point_in_plate(get_pose(self.body)[0])
                for conf in self.trajectory.path: # scripted skill
                    if steps >= horizon:
                        print('reached maximum horizon.')
                        break
                    states.append(state)
                    for i, _ in enumerate(joint_controller_hold(conf.body, conf.joints, conf.values, velocity_scale=0.1)):
                        step_simulation()
                        sim_time += sim_dt
                        if sim_time > 1/CONTROL_FREQ:
                            sim_time = 0.0
                            steps += 1
                            break # controller is reset every control loop
                    gripper_state = get_gripper_state(get_tool_pose_wrt_base(self.robot))
                    obj_pose = get_pose_wrt_base(self.robot, get_pose(self.body))
                    gripper_delta = list(map(sub, gripper_state, old_gripper_state))
                    normed_gripper_delta = list(map(truediv, gripper_delta, GRIPPER_NORM_COST))
                    old_gripper_state = gripper_state
                    obj_poses.append(obj_pose)
                    action_infos.append({"actions": np.array(normed_gripper_delta)})
                    state = get_state_wrt_base(self.robot, self.body)
                    dist = np.linalg.norm(state['obj_pose'][:2] - goal)
                    done = dist < PLATE_RADIUS
                    reward = - dist if self.dense else int(done) - 1
                    rewards.append(reward)
                    dones.append(done)
                    next_states.append(state)

                    # only end if block is near the goal
                    if self.evaluate and done:
                        break

                # evaluate TAMP trajectory
                print('using motion planner script.')
                success = is_point_in_plate(get_pose(self.body)[0])
                print('success: ', success)

                if self.evaluate:
                    return {'success': success}

                # check if block tips over
                if len(obj_poses) > 0 and obj_poses[-1][0][-1] < 0.1:
                    print('block fell.')
                    break
                last_index = len(dones) - 1
                print('trajectory length: ', last_index+1)
                # check if trajectory is too long
                first_index = 3
                if last_index >= MAX_HORIZON+first_index:
                    print('trajectory is too long.')
                    last_index = MAX_HORIZON+first_index-1
                # crop the first three indexes of the trajectory (action is always zero)
                if last_index <= first_index:
                    print('trajectory is too short.')
                    break
                pruned_states = states[first_index:last_index+1]
                pruned_action_infos = action_infos[first_index:last_index+1]
                pruned_next_states = next_states[first_index:last_index+1]
                pruned_actions = [list(dict.values())[0] for dict in pruned_action_infos]

                # if the pruned trajectories contain actions that are too large, don't save
                if any((abs(action["actions"]).max() > 1.0) for action in pruned_action_infos):
                    print('action is too large.')
                    for action in pruned_action_infos:
                            if (abs(action["actions"]).max() > 1.0):
                                print(abs(action["actions"]).max())
                    break

                if self.directory is not None:
                    expert_directory = os.path.join(self.directory, "expert")
                    if not os.path.exists(expert_directory):
                        print("Making new directory at {}".format(expert_directory))
                        os.makedirs(expert_directory)
                    t1, t2 = str(time.time()).split(".")
                    ep_directory = os.path.join(expert_directory, "ep_{}_{}".format(t1, t2))
                    assert not os.path.exists(ep_directory)
                    print("Making folder at {}".format(ep_directory))
                    os.makedirs(ep_directory)
                    state_path = os.path.join(ep_directory, "state_{}_{}.npz".format(t1, t2))
                    env_name = 'Push'
                    np.savez(
                        state_path,
                        states=np.array(pruned_states),
                        action_infos=pruned_action_infos,
                        goal=np.array([goal]),
                        env=env_name,
                    )

                pruned_rewards = rewards[first_index:last_index+1] ##
                pruned_dones = dones[first_index:last_index+1] ##
                print('dones: ', pruned_dones)


                # condition states to goal
                flattened_conditioned_pruned_states = []
                flattened_conditioned_pruned_next_states = []
                for state in pruned_states:
                    flattened_conditioned_pruned_states.append(flatten_state(condition_state(state, goal)))
                for state in pruned_next_states:
                    flattened_conditioned_pruned_next_states.append(flatten_state(condition_state(state, goal))) 

                trajectory = {
                    'states': flattened_conditioned_pruned_states,  
                    'next_states': flattened_conditioned_pruned_next_states,
                    'actions': pruned_actions,
                    'rewards': pruned_rewards,
                    'dones': pruned_dones,
                    'config': config,
                }
            else:
                if self.directory is not None: # trajectory from policy
                    trajectory = {
                        'states': states,
                        'next_states': next_states,
                        'actions': actions,
                        'rewards': rewards,
                        'dones': dones,
                    }
                else:
                    trajectory = {
                        'init': init_state,
                        'success': success,
                        'jammed': jammed_init,
                        'logs': log_dict,
                    }
            return trajectory
    def reverse(self):
        return self.trajectory.reverse()
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, get_body_name(self.robot), get_name(self.body))
    
##################################################

# TODO: make it work for side grasp as well
def get_grasp_gen(problem, collisions=False, randomize=True):
    def fn(body):
        grasps = []
        approach_vector = APPROACH_DISTANCE*get_unit_vector([0, 0, -1])
        grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), get_carry_conf('top'))
                        for g in get_top_grasps(body, grasp_length=GRASP_LENGTH))
        filtered_grasps = []
        for grasp in grasps:
            grasp_width = compute_grasp_width(problem.robot, body, grasp.value) if collisions else 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        if randomize:
            random.shuffle(filtered_grasps)
        return [(g,) for g in filtered_grasps]
    return fn

##################################################

def get_align_gen(problem, collisions=False):
    def fn(body, actual_pose, target_pose):
        #TODO: put object back to where it was
        actual_pose.assign()
        approach_vector = APPROACH_DISTANCE*get_unit_vector([0, 0, -1])
        g = get_align(body, target_pose, grasp_length=GRASP_LENGTH)
        align = Alignment('top', body, g, multiply((approach_vector, unit_quat()), g), get_carry_conf('top'))
        align.grasp_width = 0.0
        return (align,)
    return fn

##################################################
#TODO: make into generator?
# needs to pass 2 tests:
#   1. does the arm reach pose p? --> IK
#   2. are there objects along the path? --> collision check
# generates push trajectories
def get_push_gen(problem, collisions=True, max_attempts=25, ignore_traj=False, eval_dir=None, friction=False):
    robot = problem.robot
    obstacles = problem.movable if collisions else []
    def fn(*inputs):
        _, o, p0, p, g, bq, q = inputs
        blocks = list(filter(lambda b: b != o, obstacles))
        bq.assign # base conf
        set_joint_positions(robot, q.joints, q.values) # arm conf
        attachment = g.get_attachment(problem.robot)
        attachments = {attachment.child: attachment}
        #TODO: get current state, query value function of the given policy
        init_gripper_pose = get_tool_pose(robot)
        gripper_pose = multiply(p.value, invert(g.value))
        gripper_pose = align_gripper(gripper_pose, init_gripper_pose)
        if friction: # if slide, interpolate poses with variable step size and take first pose
            poses = list(interpolate_poses(init_gripper_pose, gripper_pose, pos_step_size=0.05))[:3]
            gripper_pose = poses[-1]
        arm_link = get_gripper_link(robot)
        arm_joints = get_arm_joints(robot)
        push_conf = tiago_inverse_kinematics(robot, gripper_pose)
        if (push_conf is None) or any(pairwise_collision(robot, b) for b in blocks):
            # print('Push IK failure')
            return None
        ## if model is given, will not use trajectory anyways, so return command with fake trajectory
        if ignore_traj:
            fake_conf = get_configuration(robot)
            mt = create_trajectory(robot, arm_joints, [fake_conf])
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
            return (cmd,)
        if eval_dir is not None: # for getting reachability range
            results = {}
            results['goal'] = np.linalg.norm(np.array(p.value[0][:-1]) - np.array(bq.values[:-1]))
            results['gripper'] = np.linalg.norm(np.array(gripper_pose[0][:-1]) - np.array(bq.values[:-1]))
            # create a file with a timestamp
            if not os.path.exists(eval_dir):
                print("Making new directory at {}".format(eval_dir))
                os.makedirs(eval_dir)
            t1, t2 = str(time.time()).split(".")
            eval_path = os.path.join(eval_dir, "eval_{}_{}.npz".format(t1, t2))
            np.savez(
                eval_path,
                results=results
            )
        resolutions = 0.1**np.ones(len(arm_joints))
        set_joint_positions(robot, q.joints, q.values) # default arm conf
        # get waypoints from start and end poses, and check IK & collisions through each pose
        approach_confs = []
        waypoints = []
        for pose in interpolate_poses(init_gripper_pose, gripper_pose, pos_step_size=0.05):
            conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, pose)
            if (conf is None) or any(pairwise_collision(robot, b) for b in blocks):
                # print('Approach IK failure')
                return None
            conf = get_joint_positions(robot, arm_joints)
            approach_confs.append(conf)
            waypoints.append(pose)
        # return to initial pose, then plan joint motions for each conf
        set_joint_positions(robot, q.joints, q.values) # default arm conf
        path = []
        for i, conf in enumerate(approach_confs):
            push_path = plan_joint_motion(robot, arm_joints, conf, attachments=attachments.values(),
                                                    obstacles=blocks, self_collisions=SELF_COLLISIONS,
                                                    resolutions=resolutions/2.)
            if push_path is None:
                print("No push path found.")
                # record_feasibility(0, robot, o, p, policy_dir, eval_dir)
                return None
            path += push_path
            sub_inverse_kinematics(robot, arm_joints[0], arm_link, waypoints[i])
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd,)
    return fn

# generate hook pose
#TODO: return Pose object instead of Posee
def get_hook_gen(problem, collisions=True):
    def fn(*inputs):
        o, r, p0, p1 = inputs # p0: current block pose | p1: goal block pose
        p0.assign()
        approach_vector = APPROACH_DISTANCE*get_unit_vector([0, 0, -1])
        x = np.sign(p0.value[0][0] - p1.value[0][0])
        y = np.sign(p0.value[0][1] - p1.value[0][1])
        if (x > 0):
            flip = 0
        else:
            flip = math.pi
        h = Posee(point=Point(
            x=p0.value[0][0]-HOOK_LENGTH*x,
            y=p0.value[0][1]+HOOK_WIDTH*y,
            z=p0.value[0][-1]+GRASP_LENGTH), 
            euler=Euler(0, 0, flip))
        grasp = Grasp('top', r, h, multiply((approach_vector, unit_quat()), h), get_carry_conf('top'))
        return (h,grasp) # return the pose and the grasp 
    return fn

# generate sweeping trajectory
# similar to push_gen except goal is offset by the hook distance
def get_sweep_gen(problem, collisions=True):
    robot = problem.robot
    obstacles = problem.movable if collisions else []
    def fn(*inputs):
        _, o, p1, p2, p3, g, bq, q = inputs # p2 is object goal pose, p3 is hook pose
        blocks = list(filter(lambda b: b != o and b != problem.tools[0], obstacles)) #TODO: remove tools
        bq.assign # base conf
        set_joint_positions(robot, q.joints, q.values) # arm conf
        gripper_pose = Posee(point=Point(
            x=p3[0][0] + p2.value[0][0] - p1.value[0][0],
            y=p3[0][1] + p2.value[0][1] - p1.value[0][1],
            z=p3[0][-1]),
            euler=euler_from_quat(get_tool_pose(robot)[-1])) # keep the gripper orientation
        print(gripper_pose)
        arm_link = get_gripper_link(robot)
        arm_joints = get_arm_joints(robot)
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, gripper_pose)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in blocks):
            print('Approach IK failure', approach_conf)
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
        attachment = g.get_attachment(problem.robot)
        attachments = {attachment.child: attachment}
        resolutions = 0.05**np.ones(len(arm_joints))
        set_joint_positions(robot, q.joints, q.values)
        approach_path = plan_direct_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                                obstacles=blocks, self_collisions=SELF_COLLISIONS,
                                                resolutions=resolutions/2.)
        if approach_path is None:
            print("No approach path found.")
            return None
        path = approach_path #TODO:+ push_path
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd,)
    return fn

# new ik_ir_traj_gen to account for length of the hook
def get_hook_ik_ir_traj_gen(problem, max_attempts=25, collisions=True,learned=False, teleport=False, **kwargs):
    ir_sampler = get_ir_sampler(problem, collisions=collisions, learned=learned, max_attempts=max_attempts, **kwargs)
    ik_fn = get_ik_arm_fn(problem, collisions=collisions, teleport=teleport, **kwargs)
    grasp_fn = get_grasp_gen(problem, collisions=collisions)
    def gen(*inputs):
        arm, hook, g1, block, p1, p2, p3, _ = inputs
        p3 = Pose(hook, p3) #KLUDGE: pretend like the object is where the hook moves to
        p3.assign()
        (g2,) = grasp_fn(hook)[0]
        ir_generator = ir_sampler(arm, hook, p3, g2)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p3.init:
                    print("pose not initialized")
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                print("no IR found")
                continue
            new_inputs = (arm, hook, p3, g2)
            ik_outputs = ik_fn(*(new_inputs + ir_outputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
    return gen

def get_ik_arm_fn(problem, custom_limits={}, collisions=True, teleport=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    is_ik_compiled()
        
    def fn(arm, obj, pose, grasp, base_conf):
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)} # it doesn't check for table collision?
        gripper_pose = multiply(pose.value, invert(grasp.value))
        approach_pose = multiply(pose.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot)
        arm_joints = get_arm_joints(robot)
        default_conf = grasp.carry
        pose.assign()
        base_conf.assign()
        close_gripper(robot)
        set_joint_positions(robot, arm_joints, default_conf)
        grasp_conf = tiago_inverse_kinematics(robot, gripper_pose, custom_limits=custom_limits)
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):
            if grasp_conf is not None:
               print('Grasp IK failure', grasp_conf)
            return None
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            if approach_conf is not None:
                print('Approach IK failure', approach_conf)
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.get_attachment(problem.robot)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.05**np.ones(len(arm_joints))
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  custom_limits=custom_limits, resolutions=resolutions/2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              restarts=2, iterations=25, smooth=25)
            if approach_path is None:
                print('Approach path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        arm_conf = Conf(robot, arm_joints, grasp_conf)
        return (arm_conf,cmd,)
    return fn
##################################################

# KLUDGE: always samples in the cener to align with policy goal
def get_stable_gen(problem, collisions=True, **kwargs):
    obstacles = problem.fixed if collisions else []
    def gen(body, surface):
        while True:
            if surface is None:
                break
            body_pose = sample_placement(body, surface)
            body_pose = Posee(point=Point(0., 0., body_pose[0][-1]), euler=euler_from_quat(body_pose[-1]))
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)
    return gen

##################################################

def get_tool_from_root(robot):
    root_link = link_from_name(robot, TIAGO_GRIPPER_ROOT)
    tool_link = link_from_name(robot, TIAGO_TOOL_FRAME)
    return get_relative_pose(robot, root_link, tool_link)

def iterate_approach_path(robot, gripper, pose, grasp, body=None):
    tool_from_root = get_tool_from_root(robot)
    grasp_pose = multiply(pose.value, invert(grasp.value))
    approach_pose = multiply(pose.value, invert(grasp.approach))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(gripper, multiply(tool_pose, tool_from_root))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.value))
        yield

def get_ir_sampler(problem, custom_limits={}, max_attempts=25, collisions=True, learned=True):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    gripper = problem.get_gripper()
    reachable_range = (0.45, 0.85)

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        for _ in iterate_approach_path(robot, gripper, pose, grasp, body=obj):
            if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
                print('Collision detected!')
                for b in approach_obstacles:
                    if pairwise_collision(gripper, b):
                        print("gripper + ", b)
                    else:
                        print(obj, b)
                        # wait_if_gui()
                return
        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = grasp.carry
        arm_joints = get_arm_joints(robot)
        base_joints = get_group_joints(robot, 'base')
        base_generator = uniform_pose_generator(robot, gripper_pose, reachable_range=reachable_range)
        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            count = 0
            for base_conf in islice(base_generator, max_attempts):
                count += 1
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                #print('IR attempts:', count)
                yield (bq,)
                break
            else:
                yield None
    return gen_fn

def get_ir2_sampler(problem, custom_limits={}, max_attempts=100, stream_name=None, skill_modules=None, collisions=True, stats=None, reach_dir=None, grid_search=True, viz=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []

    def gen_fn(arm, obj, pose, end_pose, grasp):
        pose.assign()
        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = grasp.carry
        arm_joints = get_arm_joints(robot)
        base_joints = get_group_joints(robot, 'base')
        skill_model = None
        if skill_modules is not None:
            # KLUDGE: read from file to check if heuristic failed
            try:
                with open(os.path.join(TEMP_SKILLS_DIR,"heuristic.txt"), "r") as f:
                    heuristic_failed = f.read() == "failed"
            except Exception:
                print('could not load heuristic.txt')
                heuristic_failed = False
            # KLDUGE: read from file to check which value function to use
            if not heuristic_failed:
                try:
                    with open(os.path.join(TEMP_SKILLS_DIR,"matching_streams.txt"), "r") as f:
                        stream_pairs = f.read()
                except Exception:
                    print('could not load matching_streams.txt')
                    stream_pairs = None
                if stream_pairs is not None:
                    # filter the pairs to ones with the stream corresponding to this sampler
                    skill_model = recover_skill_model_from_stream_pairs(stream_pairs, skill_modules, stream_name)
              # skill_model = skill_modules['push']['plan_push_motion'].vf.v # expected
        if skill_model is not None:
           base_generator = learned_pose_generator(robot, obj, gripper_pose, end_pose.value, skill_model, stats, reach_dir, grid_search=grid_search, viz=viz)
        else:
            base_generator = custom_pose_generator(robot, gripper_pose, end_pose.value)
        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            count = 0
            for base_conf in islice(base_generator, max_attempts):
                count += 1
                if not all_between(lower_limits, base_conf, upper_limits):
                    print('out of bounds.')
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                yield (bq,)
                break
            else:
                yield None
    return gen_fn

##################################################

def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    is_ik_compiled()
        
    def fn(arm, obj, pose, grasp, base_conf):
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose.value, invert(grasp.value))
        approach_pose = multiply(pose.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot)
        arm_joints = get_arm_joints(robot)
        default_conf = grasp.carry
        pose.assign()
        base_conf.assign()
        open_gripper(robot)
        set_joint_positions(robot, arm_joints, default_conf)
        grasp_conf = tiago_inverse_kinematics(robot, gripper_pose, custom_limits=custom_limits)
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):
            if grasp_conf is not None:
               print('Grasp IK failure', grasp_conf)
            return None
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            if approach_conf is not None:
                print('Approach IK failure', approach_conf)
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.get_attachment(problem.robot)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.05**np.ones(len(arm_joints))
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  custom_limits=custom_limits, resolutions=resolutions/2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              restarts=2, iterations=25, smooth=25)
            if approach_path is None:
                print('Approach path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd,)
    return fn

##################################################

def get_ik_ir_gen(problem, max_attempts=25, collisions=True,learned=False, teleport=False, **kwargs):
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=max_attempts, **kwargs)
    ik_fn = get_ik_fn(problem, collisions=collisions, teleport=teleport, **kwargs)
    def gen(*inputs):
        b, a, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
    return gen

##################################################

def get_motion_gen(problem, collisions=True, teleport=False):
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    def fn(bq1, bq2, fluents=[]):
        saver.restore()
        bq1.assign()
        if teleport:
            path = [bq1, bq2]
        else:
            raw_path = plan_joint_motion(robot, bq2.joints, bq2.values, attachments=[],
                                         obstacles=obstacles, custom_limits=[], self_collisions=SELF_COLLISIONS,
                                         restarts=4, iterations=50, smooth=50)
            if raw_path is None:
                print('Failed motion plan!')
                return None
            path = [Conf(robot, bq2.joints, q) for q in raw_path]
        bt = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
        return (cmd,)
    return fn

##################################################

# returns the arm configuration and the trajectory
# KLUDGE: trajectory ignores obstacles because the table is considered as one
def get_ik_traj_fn(problem, custom_limits={}, collisions=True, teleport=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    is_ik_compiled()
        
    def fn(arm, obj, pose1, pose2, grasp, base_conf):
        gripper_pose = Posee(point=pose1.value[0] + grasp.value[0], euler=euler_from_quat(grasp.value[1])) # grasp value is in world frame
        approach_pose = multiply(pose1.value, invert(grasp.approach)) # approach value is in object pose frame
        approach_pose = Posee(point=approach_pose[0] + grasp.value[0], euler=euler_from_quat(grasp.value[1])) # grasp value is in world frame
        arm_link = get_gripper_link(robot)
        arm_joints = get_arm_joints(robot)
        default_conf = grasp.carry
        pose1.assign()
        base_conf.assign()
        close_gripper(robot)
        set_joint_positions(robot, arm_joints, default_conf)
        grasp_conf = tiago_inverse_kinematics(robot, gripper_pose, custom_limits=custom_limits)
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):
            if grasp_conf is not None:
               print('Grasp IK failure', grasp_conf)
            return None
        attachment = grasp.get_attachment(problem.robot)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, grasp_conf]
        else:
            resolutions = 0.05**np.ones(len(arm_joints))
            set_joint_positions(robot, arm_joints, default_conf)
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  self_collisions=SELF_COLLISIONS, custom_limits=custom_limits,
                                                  resolutions=resolutions/2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            path = grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        arm_conf = Conf(robot, arm_joints, grasp_conf)
        return (arm_conf,cmd,)
    return fn

# returns the base configuration, arm config and arm trajectory
def get_ik_ir_traj_gen(problem, stream_name=None, skill_modules=None, max_attempts=100, collisions=True, teleport=False, stats=None, reach_dir=None, grid_search=True, viz=False, **kwargs):
    ir_sampler = get_ir2_sampler(problem, max_attempts=max_attempts, stream_name=stream_name, skill_modules=skill_modules, stats=stats, reach_dir=reach_dir, grid_search=grid_search, viz=viz, **kwargs)
    ik_fn = get_ik_traj_fn(problem, collisions=collisions, teleport=teleport, **kwargs)
    def gen(*inputs):
        _, _, p1, _, _ = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p1.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            # print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
    return gen

##################################################

# get arm config without the path
def get_ik_only_fn(problem, custom_limits={}, collisions=True, teleport=False):
    robot = problem.robot
    # obstacles = problem.fixed if collisions else []
    is_ik_compiled()
        
    def fn(_, obj, pose, grasp, base_conf):
        # approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose.value, invert(grasp.value))
        # approach_pose = multiply(pose.value, invert(grasp.approach))
        # arm_link = get_gripper_link(robot)
        arm_joints = get_arm_joints(robot)
        default_conf = grasp.carry
        pose.assign()
        base_conf.assign()
        open_gripper(robot)
        set_joint_positions(robot, arm_joints, default_conf)
        grasp_conf = tiago_inverse_kinematics(robot, gripper_pose, custom_limits=custom_limits)
        arm_conf = Conf(robot, arm_joints, grasp_conf)
        return (arm_conf,)
    return fn

# get base config based on arm config, return both
def get_ik_ir_only_gen(problem, max_attempts=25, collisions=True,learned=False, teleport=False, **kwargs):
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=max_attempts, **kwargs)
    ik_fn = get_ik_only_fn(problem, collisions=collisions, teleport=teleport, **kwargs)
    def gen(*inputs):
        b, a, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
    return gen

##################################################

def control_commands(commands, **kwargs):
    #wait_if_gui('Control?')
    disable_real_time()
    enable_gravity()
    trajectories = []
    for i, command in enumerate(commands):
        # print(i, command)
        trajectories.append(command.control(*kwargs))
    return trajectories

def apply_commands(state, commands, time_step=None, pause=False, **kwargs):
    #wait_if_gui('Apply?')
    for i, command in enumerate(commands):
        print(i, command)
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                wait_if_gui('Command {}, Step {}) Next?'.format(i, j))
            else:
                wait_for_duration(time_step)
        if pause:
            wait_if_gui()

def get_feasibility_estimate(model, state):
    return model.vf.query(state, device=model.device)

def get_state_from_base_values(base_values, robot, stats, start_body, gripper_pose, goal_pose):
    bq = Conf(robot, get_group_joints(robot, 'base'), base_values) # set the base conf
    bq.assign()
    state = get_state_wrt_base(robot, start_body)
    state["tool_pose"] = pose2d_from_pose(get_pose_wrt_base(robot, gripper_pose)) # set tool_pose to gripper_pose
    state = flatten_state(condition_state(state, get_goal_wrt_base(robot, goal_pose)))
    if stats is not None:
        state = normalize_state(state, stats)
    return state

# check if within the annulus first, if not then return zero
def is_within_annulus(params, gripper_pose, goal_pose):
    return True if point_in_annulus(
        params[0], params[1], gripper_pose[0][0], gripper_pose[0][1], GRIPPER_REACHABLE_RANGE[0], GRIPPER_REACHABLE_RANGE[1]
        ) and point_in_annulus(
            params[0], params[1], goal_pose[0][0], goal_pose[0][1], BLOCK_REACHABLE_RANGE[0], BLOCK_REACHABLE_RANGE[1]
            ) else False

def get_base_values(params, gripper_pose):
    (radius, theta, phi) = params
    x, y = radius*unit_from_theta(theta) + gripper_pose[0][:2] #pose2d_from_pose(get_pose(start_pose))[:2]
    return (x, y, phi)

# samples base pose from learned value function
def learned_pose_generator(robot, start_body, gripper_pose, goal_pose, model, stats, reach_dir, grid_search, viz, max_attempts=100):
    with open(os.path.join(TEMP_SKILLS_DIR,"attempts.txt"), "r") as f:
        attempts = int(f.read())
    if attempts == 0: # if first attempt, run optimization
        x = (-1.2, 1.2)
        y = x
        theta = (0., math.pi) #-math.pi/2
        if reach_dir is not None: # for reachability test
            # compare with uniform
            base_values = sample_reachable_base(robot, point_from_pose(gripper_pose), reachable_range=GRIPPER_REACHABLE_RANGE)
            goal_is_reachable = is_reachable(base_values, goal_pose, BLOCK_REACHABLE_RANGE)
            print('goal block reachability: ', goal_is_reachable)
            results = {}
            if not os.path.exists(reach_dir):
                print("Making new directory at {}".format(reach_dir))
                os.makedirs(reach_dir)
            results['uniform'] = int(goal_is_reachable)        
        if grid_search: # OPTION 1: GRID SEARCH
            min_x, max_x, min_y, max_y = get_bounding_box(gripper_pose[0], goal_pose[0], 
                                                            GRIPPER_REACHABLE_RANGE, BLOCK_REACHABLE_RANGE)
            grid_num = 15
            param1_range = np.linspace(start=min_x, stop=max_x, num=grid_num)
            param2_range = np.linspace(start=min_y, stop=max_y, num=grid_num)
            param3_range = np.linspace(start=theta[0], stop=theta[-1], num=len(theta))
            best_params = None
            grid_shape = (param1_range.size, param2_range.size, param3_range.size)
            grid_data = np.zeros(grid_shape)
            for i, param1 in enumerate(param1_range):
                for j, param2 in enumerate(param2_range):
                    for k, param3 in enumerate(param3_range):
                        params = (param1, param2, param3)
                        if is_within_annulus(params, gripper_pose, goal_pose):
                            state = get_state_from_base_values(params, robot, stats, start_body, gripper_pose, goal_pose)
                            metric = get_feasibility_estimate(model, state)
                        else:
                            metric = np.nan
                        grid_data[i, j, k] = metric
            # replace nan's with the worst 2d value for visualization
            grid_data_2d = np.nanmax(grid_data, 2)
            worst_value_2d = np.nanmin(grid_data_2d)
            grid_data = np.nan_to_num(grid_data, nan=worst_value_2d)
            sorted_indices = sort_3d_array_indices_desc(grid_data)
            indices = sorted_indices.pop(0)
            best_params = (param1_range[indices[0]], param2_range[indices[1]], param3_range[indices[2]])
            best_params = maybe_flip_phi(best_params, start_body)
            if viz:
                grid = np.max(grid_data, 2) # only x and y
                show_heatmap(grid)
            learned_base_values = best_params
        else:
            from scipy.optimize import minimize
            method = 'Nelder-Mead' #'Nelder-Mead' #'L-BFGS-B' #'Powell'
            # bounds = ((min_x, max_x), (min_y, max_y), (-math.pi, math.pi))
            bounds = (GRIPPER_REACHABLE_RANGE, (-math.pi, math.pi), (-math.pi, math.pi))
            initial_guess = [mean(bound) for bound in bounds]
            def objective(params, robot, model, start_pose, gripper_pose, goal_pose):
                base_values = get_base_values(params, gripper_pose)
                state = get_state_from_base_values(base_values, robot, stats, start_body, gripper_pose, goal_pose)
                return -get_feasibility_estimate(model, state)
            from functools import partial
            partial_obj = partial(
                objective, 
                robot=robot,
                model=model,
                start_pose=start_body,
                gripper_pose=gripper_pose,
                goal_pose=goal_pose
            )
            result = minimize(
                partial_obj,
                initial_guess,
                method=method,
                bounds=bounds,
                options={'xatol': 1e-6, 'disp': False}
            )
            (radius, theta, phi) = result.x # convert back to world pose
            x, y = radius*unit_from_theta(theta) + gripper_pose[0][:2]
            learned_base_values = (x, y, phi)
            learned_base_values = maybe_flip_phi(learned_base_values, start_body)
            wait_if_gui()
                
        # check that the base is in the reachable zone
        goal_is_reachable = is_reachable(learned_base_values, goal_pose, BLOCK_REACHABLE_RANGE) # epsilon=0.07

        # write result to write
        with open(os.path.join(TEMP_SKILLS_DIR,"learned_base_values.txt"), "w") as f:
            for base_value in learned_base_values:
                f.write(f"{base_value}\n")
        if reach_dir is not None: # for reachability test
            results['policy'] = int(goal_is_reachable)
            t1, t2 = str(time.time()).split(".")
            col_path = os.path.join(reach_dir, "reachable_{}_{}.npz".format(t1, t2))
            np.savez(
                col_path,
                results=results
            )
            time.sleep(0.2) # KLUDGE: give time before program crashes
            yield None
    else: # if not first attempt, read from file
        with open(os.path.join(TEMP_SKILLS_DIR,"learned_base_values.txt"), "r") as f:
            learned_base_values = tuple(float(line.strip()) for line in f)
    while True:
        if max_attempts <= attempts: # revert to baseline
            attempts = 0
            with open(os.path.join(TEMP_SKILLS_DIR,"heuristic.txt"), "w") as f:
                f.write("failed")
            with open(os.path.join(TEMP_SKILLS_DIR,"attempts.txt"), "w") as f:
                f.write("0")
            with open(os.path.join(TEMP_SKILLS_DIR,"learned_base_values.txt"), "w") as f:
                f.write("")
            print('heuristic failed.')
            break
        attempts += 1
        reachable_base_values = None
        for _ in range(HEURISTIC_ATTEMPTS): # sample base pose within the reachable zone
            base_values = perturb_base(learned_base_values, perturb_range=(0.0, MAX_PERTURBATION))
            if point_in_annulus(
                base_values[0], base_values[1], gripper_pose[0][0], gripper_pose[0][1], 
                GRIPPER_REACHABLE_RANGE[0], GRIPPER_REACHABLE_RANGE[1]
                ) and point_in_annulus(
                    base_values[0], base_values[1], goal_pose[0][0], goal_pose[0][1], 
                    BLOCK_REACHABLE_RANGE[0], BLOCK_REACHABLE_RANGE[1]
                    ):
                reachable_base_values = base_values
                break
        if reachable_base_values is not None:
            with open(os.path.join(TEMP_SKILLS_DIR,"attempts.txt"), "w") as f:
                f.write(str(attempts))
            yield base_values
        else:
            with open(os.path.join(TEMP_SKILLS_DIR,"heuristic.txt"), "w") as f:
                f.write("failed")
            with open(os.path.join(TEMP_SKILLS_DIR,"attempts.txt"), "w") as f:
                f.write("0")
            with open(os.path.join(TEMP_SKILLS_DIR,"learned_base_values.txt"), "w") as f:
                f.write("")
            print('heuristic failed (not reachable).')
            break
    while True:
        base_values = sample_reachable_base2(robot, point_from_pose(gripper_pose), point_from_pose(goal_pose))
        if base_values is None:
            break
        yield base_values

def list_to_numpy(dict_of_list):
    dic = OrderedDict()
    for k in dict_of_list:
        dic[k] = np.array(dict_of_list[k])
    return dic

def pad_list_to_length(input_list, desired_length, last_element=None):
    if last_element is None:
        last_element = input_list[-1]
    # assert(len(input_list[-1]) == len(last_element))
    while len(input_list) < desired_length:
        input_list.append(last_element)
    return input_list

def pad_list_to_length_with_zeros(input_list, desired_length):
    last_element = np.zeros(input_list[-1].shape)
    while len(input_list) < desired_length:
        input_list.append(last_element)
    return input_list