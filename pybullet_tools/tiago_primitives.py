from itertools import islice
import random
import time

import numpy as np

from .ikfast.tiago.ik import is_ik_compiled, tiago_inverse_kinematics
from .pr2_primitives import Command, Commands, Conf, Pose, State, Trajectory, create_trajectory
from .tiago_utils import TIAGO_GRIPPER_ROOT, TIAGO_GROUPS, TIAGO_TOOL_FRAME, TOP_HOLDING_LEFT_ARM, compute_grasp_width, get_arm_joints, get_gripper_joints, get_gripper_link, get_group_joints, get_top_grasps, open_gripper
from .utils import Attachment, BodySaver, Pose2d, add_fixed_constraint, all_between, base_values_from_pose, create_attachment, get_body_name, get_custom_limits, get_extend_fn, get_joint_positions, get_link_pose, get_min_limit, get_name, get_pose, get_relative_pose, get_unit_vector, interpolate_poses, invert, is_placement, joint_controller_hold, joints_from_names, link_from_name, multiply, pairwise_collision, plan_base_motion, plan_direct_joint_motion, plan_joint_motion, pose_from_base_values, pose_from_pose2d, remove_fixed_constraint, sample_placement, set_joint_positions, set_pose, step_simulation, sub_inverse_kinematics, uniform_pose_generator, unit_quat, wait_if_gui

BASE_EXTENT = 3.5 # 2.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False

# need own grasp class to specify tool link
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
            yield

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
        grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
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

def get_stable_gen(problem, collisions=True, **kwargs):
    obstacles = problem.fixed if collisions else []
    def gen(body, surface):
        while True:
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