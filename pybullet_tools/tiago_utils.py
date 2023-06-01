import math
import numpy as np
from .pr2_utils import get_gripper_joints,close_until_collision, set_joint_position, get_max_limit, \    get_min_limit, joints_from_names
from .utils import PIPI, TRANSPARENT, approximate_as_prism, clone_body, get_joint_positions, Euler, Pose, get_link_pose, get_link_subtree, link_from_name, multiply, point_from_pose, set_all_color, set_joint_positions, set_pose, unit_pose

TIAGO_URDF = "models/tiago_description/tiago.urdf"

EXTENDED_ARM = [0., 0., 0., 0., 0., 0., 0.]
TUCKED_ARM = [1.5, 0.59, 0.06, 1.0, -1.7, 0., 0.] # TODO: get actual values

TIAGO_GROUPS = {
    'base': ['x', 'y', 'theta'],
    'torso': ['torso_lift_joint'],
    'head': ['head_1_joint', 'head_2_joint'], # pan, tilt
    'arm': ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', # shoulder x3
            'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint'], # elbow, wrist x3
    'gripper': ['gripper_right_finger_joint', 'gripper_left_finger_joint'],
}

GRASP_LENGTH = 0.
MAX_GRASP_WIDTH = np.inf
TOOL_POSE = Pose(euler=Euler(pitch=PI))
#TODO: find actual values
TOP_HOLDING_LEFT_ARM = [0.8, 0.74313199, 1.5, -1.46688405, 0.94223229, 1.75442826, 2.22254125]
SIDE_HOLDING_LEFT_ARM = [0.39277395, 0.33330058, 0., -1.52238431, 2.72170996, -1.21946936, -2.98914779]

TIAGO_LEFT_CARRY_CONFS = {
    'top': TOP_HOLDING_LEFT_ARM,
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

def get_gripper_joints(robot):
    return get_group_joints(robot, 'gripper')

def get_carry_conf(grasp_type):
    return TIAGO_LEFT_CARRY_CONFS[grasp_type]

def set_arm_conf(robot, conf):
    set_joint_positions(robot, get_arm_joints(robot), conf)

def get_top_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                   max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH):
    # TODO: rename the box grasps
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

def compute_grasp_width(robot, body, grasp_pose, **kwargs):
    tool_link = get_gripper_link(robot)
    tool_pose = get_link_pose(robot, tool_link)
    body_pose = multiply(tool_pose, grasp_pose)
    set_pose(body, body_pose)
    gripper_joints = get_gripper_joints(robot)
    return close_until_collision(robot, gripper_joints, bodies=[body], **kwargs)

def create_gripper(robot, visual=True):
    link_name = TIAGO_GRIPPER_ROOT
    # gripper = load_pybullet(os.path.join(get_data_path(), 'pr2_gripper.urdf'))
    # gripper = load_pybullet(os.path.join(get_models_path(), 'pr2_description/pr2_l_gripper.urdf'), fixed_base=False)
    # pybullet.error: Error receiving visual shape info for the DRAKE_PR2
    links = get_link_subtree(robot, link_from_name(robot, link_name))
    gripper = clone_body(robot, links=links, visual=False, collision=False)  # TODO: collision to True
    if not visual:
        set_all_color(robot, TRANSPARENT)
    return gripper