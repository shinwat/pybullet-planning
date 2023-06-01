from .pr2_utils import get_gripper_joints, set_joint_position, get_max_limit, \
    get_min_limit, joints_from_names
from .utils import PI

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

def open_gripper(robot):
    for joint in joints_from_names(robot, TIAGO_GROUPS['gripper']):
        set_joint_position(robot, joint, get_max_limit(robot, joint))

def close_gripper(robot):
    for joint in joints_from_names(robot, TIAGO_GROUPS['arm']):
        set_joint_position(robot, joint, get_min_limit(robot, joint))