import copy
import random
import time
from itertools import islice, count

import numpy as np

from .ikfast.tiago.ik import get_tool_pose, is_ik_compiled, tiago_inverse_kinematics
from .pr2_primitives import State, Trajectory, create_trajectory
from .tiago_utils import TIAGO_GRIPPER_ROOT, TIAGO_GROUPS, TIAGO_TOOL_FRAME, TOOL_POSE, TOP_HOLDING_LEFT_ARM, align_gripper, compute_grasp_width, get_align, get_arm_conf, get_arm_joints, get_carry_conf, get_gripper_joints, get_gripper_link, get_group_joints, get_midpoint_pose, get_top_grasps, open_gripper
from .utils import Attachment, BodySaver, Pose2d, add_fixed_constraint, all_between, approximate_as_prism, base_values_from_pose, create_attachment, disable_real_time, enable_gravity, enable_real_time, flatten_links, get_body_name, get_configuration, get_custom_limits, get_extend_fn, get_joint_positions, get_link_pose, get_min_limit, get_moving_links, get_name, get_pose, get_relative_pose, get_unit_vector, interpolate_poses, inverse_kinematics, invert, is_placement, joint_controller_hold, joints_from_names, link_from_name, multiply, pairwise_collision, plan_base_motion, plan_direct_joint_motion, plan_joint_motion, point_from_pose, pose_from_base_values, pose_from_pose2d, remove_fixed_constraint, sample_placement, set_base_values, set_joint_positions, set_pose, step_simulation, sub_inverse_kinematics, uniform_pose_generator, unit_pose, unit_quat, wait_for_duration, wait_if_gui

BASE_EXTENT = 3.5 # 2.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False

##################################################

class Pose(object):
    num = count()
    #def __init__(self, position, orientation):
    #    self.position = position
    #    self.orientation = orientation
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
        align = Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), get_carry_conf('top'))
        align.grasp_width = 0.0
        return (align,)
    return fn

##################################################
#TODO: make into generator?
# needs to pass 2 tests:
#   1. does the arm reach pose p? --> IK
#   2. are there objects along the path? --> collision check
# generates push trajectories
def get_push_gen(problem, collisions=True, max_attempts=25):
    robot = problem.robot
    obstacles = problem.movable if collisions else []
    def fn(*inputs):
        _, o, p0, p, g, bq, q = inputs
        blocks = list(filter(lambda b: b != o, obstacles))
        bq.assign # base conf
        set_joint_positions(robot, q.joints, q.values) # arm conf
        init_gripper_pose = get_tool_pose(robot)
        gripper_pose = multiply(p.value, invert(g.value))
        gripper_orientation = multiply(p0.value, invert(g.value))
        gripper_pose = align_gripper(gripper_pose, gripper_orientation)
        approach_pose = get_midpoint_pose(init_gripper_pose, gripper_pose)
        arm_link = get_gripper_link(robot)
        arm_joints = get_arm_joints(robot)
        push_conf = tiago_inverse_kinematics(robot, gripper_pose)
        if (push_conf is None) or any(pairwise_collision(robot, b) for b in blocks):
            print('Push IK failure')
            return None
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in blocks):
            print('Approach IK failure')
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
        # print("Tool pose: ", get_tool_pose(robot))
        # set_joint_positions(robot, q.joints, q.values) # default arm conf
        # print("Tool pose: ", get_tool_pose(robot))
        attachment = g.get_attachment(problem.robot)
        attachments = {attachment.child: attachment}
        resolutions = 0.05**np.ones(len(arm_joints))
        push_path = plan_direct_joint_motion(robot, arm_joints, push_conf, attachments=attachments.values(),
                                                obstacles=blocks, self_collisions=SELF_COLLISIONS,
                                                resolutions=resolutions/2.)
        if push_path is None:
            print("No push path found.")
            return None
        set_joint_positions(robot, q.joints, q.values) # default arm conf
        approach_path = plan_direct_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                                obstacles=blocks, self_collisions=SELF_COLLISIONS,
                                                resolutions=resolutions/2.)
        if approach_path is None:
            print("No approach path found.")
            return None
        path = approach_path + push_path #
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd,)
    return fn

##################################################

def get_stable_gen(problem, collisions=True, **kwargs):
    obstacles = problem.fixed if collisions else []
    def gen(body, surface):
        while True:
            if surface is None:
                break
            body_pose = sample_placement(body, surface)
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

#TODO: implement learned sampler for tiago
def get_ir_sampler(problem, custom_limits={}, max_attempts=25, collisions=True, learned=True):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    gripper = problem.get_gripper()

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
                        wait_if_gui()
                return
        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = grasp.carry
        arm_joints = get_arm_joints(robot)
        base_joints = get_group_joints(robot, 'base')
        # if learned:
            # base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        # else:
        base_generator = uniform_pose_generator(robot, gripper_pose)
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
                print('IR attempts:', count)
                yield (bq,)
                break
            else:
                yield None
    return gen_fn

##################################################

def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    if is_ik_compiled():
        print('Using ikfast for inverse kinematics')
    else:
        print('Using pybullet for inverse kinematics')
        
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
def get_ik_traj_fn(problem, custom_limits={}, collisions=True, teleport=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    if is_ik_compiled():
        print('Using ikfast for inverse kinematics')
    else:
        print('Using pybullet for inverse kinematics')
        
    def fn(arm, obj, pose1, pose2, grasp, base_conf):
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose1.value, invert(grasp.value))
        approach_pose = multiply(pose1.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot)
        arm_joints = get_arm_joints(robot)
        default_conf = grasp.carry
        pose1.assign()
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
        arm_conf = Conf(robot, arm_joints, grasp_conf)
        return (arm_conf,cmd,)
    return fn

# returns the base configuration, arm config and arm trajectory
def get_ik_ir_traj_gen(problem, max_attempts=25, collisions=True,learned=False, teleport=False, **kwargs):
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=max_attempts, **kwargs)
    ik_fn = get_ik_traj_fn(problem, collisions=collisions, teleport=teleport, **kwargs)
    def gen(*inputs):
        b, a, p1, _, g = inputs
        ir_generator = ir_sampler(b, a, p1, g)
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
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
    return gen

##################################################

# get arm config without the path
def get_ik_only_fn(problem, custom_limits={}, collisions=True, teleport=False):
    robot = problem.robot
    # obstacles = problem.fixed if collisions else []
    if is_ik_compiled():
        print('Using ikfast for inverse kinematics')
    else:
        print('Using pybullet for inverse kinematics')
        
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
    wait_if_gui('Control?')
    disable_real_time()
    enable_gravity()
    for i, command in enumerate(commands):
        print(i, command)
        command.control(*kwargs)

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