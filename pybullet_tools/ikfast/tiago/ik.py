import random

from ...tiago_utils import TIAGO_TOOL_FRAME, get_arm_joints, get_gripper_link, get_torso_arm_joints

from ..utils import get_ik_limits, compute_forward_kinematics, compute_inverse_kinematics, select_solution, \
    USE_ALL, USE_CURRENT
from ...utils import get_relative_pose, multiply, get_link_pose, link_from_name, get_joint_positions, \
    joint_from_name, invert, get_custom_limits, all_between, sub_inverse_kinematics, set_joint_positions, \
    get_joint_positions, pairwise_collision
from ...ikfast.utils import IKFastInfo
#from ...ikfast.ikfast import closest_inverse_kinematics # TODO: use these functions instead

# TODO: deprecate

IK_FRAME = 'gripper_tool_frame' #gripper_tool_frame, gripper_link
BASE_FRAME = 'base_link'
TORSO_FRAME = 'torso_lift_link'

# TORSO_JOINT = 'torso_lift_joint' #TODO: make new ikfast solver with torso included
UPPER_JOINT = 'arm_3_joint' # Third arm joint (elbow)


#####################################

PR2_URDF = "models/tiago_description/tiago.urdf"

TIAGO_INFO = IKFastInfo(module_name='tiago.ikfast_tiago_arm', base_link=TORSO_FRAME,
                             ee_link=IK_FRAME, free_joints=[UPPER_JOINT]) #TORSO_JOINT


#####################################

def get_tool_pose(robot):
    from .ikfast_tiago_arm import get_fk
    ik_joints = get_arm_joints(robot) #TODO: should be get_torso_arm_joints
    conf = get_joint_positions(robot, ik_joints)
    assert len(conf) == 7 #TODO: should be 8 with torso
    torso_from_tool = compute_forward_kinematics(get_fk, conf)
    base_from_torso = get_relative_pose(robot, link_from_name(robot, TORSO_FRAME), link_from_name(robot, BASE_FRAME))# static transform from torso to 
    base_from_tool = multiply(base_from_torso, torso_from_tool)
    #quat = quat if quat.real >= 0 else -quat  # solves q and -q being same rotation
    world_from_base = get_link_pose(robot, link_from_name(robot, BASE_FRAME))
    return multiply(world_from_base, base_from_tool)

#####################################

def is_ik_compiled():
    try:
        from .ikfast_tiago_arm import get_ik
        return True
    except ImportError:
        return False

def get_ik_generator(robot, ik_pose, torso_limits=USE_ALL, upper_limits=USE_ALL, custom_limits={}):
    from .ikfast_tiago_arm import get_ik
    world_from_base = get_link_pose(robot, link_from_name(robot, TORSO_FRAME))
    base_from_ik = multiply(invert(world_from_base), ik_pose)
    sampled_joints = [joint_from_name(robot, name) for name in [UPPER_JOINT]]
    sampled_limits = [get_ik_limits(robot, joint, limits) for joint, limits in zip(sampled_joints, [torso_limits, upper_limits])]
    arm_joints = get_arm_joints(robot) #get_torso_arm_joints

    min_limits, max_limits = get_custom_limits(robot, arm_joints, custom_limits)
    while True:
        sampled_values = [random.uniform(*limits) for limits in sampled_limits]
        confs = compute_inverse_kinematics(get_ik, base_from_ik, sampled_values)
        solutions = [q for q in confs if all_between(min_limits, q, max_limits)]
        # TODO: return just the closest solution
        # print(len(confs), len(solutions))
        yield solutions
        if all(lower == upper for lower, upper in sampled_limits):
            break

def get_tool_from_ik(robot):
    world_from_tool = get_link_pose(robot, link_from_name(robot, TIAGO_TOOL_FRAME))
    world_from_ik = get_link_pose(robot, link_from_name(robot, IK_FRAME))
    return multiply(invert(world_from_tool), world_from_ik)

def sample_tool_ik(robot, tool_pose, nearby_conf=USE_ALL, max_attempts=25, **kwargs):
    ik_pose = multiply(tool_pose, get_tool_from_ik(robot))
    generator = get_ik_generator(robot, ik_pose, **kwargs)
    arm_joints = get_arm_joints(robot) #get_torso_arm_joints
    for _ in range(max_attempts):
        try:
            solutions = next(generator)
            # TODO: sort by distance from the current solution when attempting?
            if solutions:
                return select_solution(robot, arm_joints, solutions, nearby_conf=nearby_conf)
        except StopIteration:
            break
    return None

def tiago_inverse_kinematics(robot, gripper_pose, obstacles=[], custom_limits={}, **kwargs):
    arm_link = get_gripper_link(robot)
    arm_joints = get_arm_joints(robot)
    if is_ik_compiled():
        ik_joints = get_arm_joints(robot) #get_torso_arm_joints
        torso_arm_conf = sample_tool_ik(robot, gripper_pose, custom_limits=custom_limits,
                                        torso_limits=USE_CURRENT, **kwargs)
        if torso_arm_conf is None:
            return None
        set_joint_positions(robot, ik_joints, torso_arm_conf)
    else:
        arm_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, gripper_pose, custom_limits=custom_limits)
        if arm_conf is None:
            return None
    if any(pairwise_collision(robot, b) for b in obstacles):
        return None
    return get_joint_positions(robot, arm_joints)
