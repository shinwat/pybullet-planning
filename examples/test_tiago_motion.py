#!/usr/bin/env python

from __future__ import print_function

import pybullet as p
import time

from pybullet_tools.pr2_primitives import Pose, create_trajectory
from pybullet_tools.tiago_problems import create_tiago
from pybullet_tools.pr2_problems import create_floor, create_table
from pybullet_tools.tiago_utils import TIAGO_TOOL_FRAME, TIAGO_URDF, TUCKED_ARM, EXTENDED_ARM, TIAGO_GROUPS, get_group_conf, get_group_joints, open_gripper, set_group_conf
from pybullet_tools.utils import joint_controller_hold, pose_from_pose2d, set_base_values, joint_from_name, quat_from_euler, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, wait_if_gui, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_if_gui, load_pybullet, set_quat, Euler, PI, RED, add_line, \
    wait_for_duration, LockRenderer

SLEEP = 0.01

##################################### TEST BASE #####################################
def test_base_motion(tiago, base_start, base_goal, obstacles=[]):
    #disabled_collisions = get_disabled_collisions(pr2)
    set_base_values(tiago, base_start)
    wait_if_gui('Plan Base?')
    base_limits = ((-2.5, -2.5), (2.5, 2.5))
    with LockRenderer(lock=False):
        base_path = plan_base_motion(tiago, base_goal, base_limits, obstacles=obstacles)
    if base_path is None:
        print('Unable to find a base path')
        return
    print(len(base_path))
    for bq in base_path:
        set_base_values(tiago, bq)
        if SLEEP is None:
            wait_if_gui('Continue?')
        else:
            wait_for_duration(SLEEP)

# test to check simulation ability for motion_gen
def test_base_control(tiago, base_start, base_goal, obstacles=[]):
    # setup
    set_base_values(tiago, base_start) # undo last test
    base_joints = [joint_from_name(tiago, name) for name in TIAGO_GROUPS['base']]
    pose_start = Pose(tiago, get_pose(tiago)) # recover robot pose from start
    pose_start.assign
    base_goal = base_goal[:len(base_joints)]
    pose_goal = Pose(tiago, pose_from_pose2d(base_goal))
    goal_conf = pose_goal.to_base_conf()
    ######
    print(pose_start)
    print(goal_conf.values)
    wait_if_gui('Plan Base?')
    with LockRenderer(lock=False):
        base_path = plan_joint_motion(tiago, goal_conf.joints, goal_conf.values, attachments=[],
                                obstacles=obstacles, restarts=4, iterations=50, smooth=50)
    if base_path is None:
        print('Unable to find a base path')
        return
    print(len(base_path))
    mt = create_trajectory(tiago, base_joints, base_path)
    real_time = False
    enable_gravity()
    p.setRealTimeSimulation(real_time)
    for conf in mt.path:
        for _ in joint_controller_hold(tiago, base_joints, conf.values):
            if not real_time:
                p.stepSimulation()

##################################### TEST ARM #####################################

def test_arm_motion(tiago, arm_joints, arm_goal):
    # TODO: get_disabled_collisions(tiago) instead of setting self collision to false
    wait_if_gui('Plan Arm?')
    with LockRenderer(lock=False):
        arm_path = plan_joint_motion(tiago, arm_joints, arm_goal, self_collisions=False)
    if arm_path is None:
        print('Unable to find an arm path')
        return
    print(len(arm_path))
    for q in arm_path:
        set_joint_positions(tiago, arm_joints, q)
        if SLEEP is None:
            wait_if_gui('Continue?')
        else:
            wait_for_duration(SLEEP)

def test_arm_control(tiago, left_joints, arm_start):
    wait_if_gui('Control Arm?')
    real_time = True
    enable_gravity()
    p.setRealTimeSimulation(real_time)
    for _ in joint_controller_hold(tiago, left_joints, arm_start):
        if not real_time:
            p.stepSimulation()

def test_ikfast(tiago):
    wait_if_gui('Test IK?')
    from pybullet_tools.ikfast.tiago.ik import get_tool_pose, get_ik_generator
    left_joints = joints_from_names(tiago, TIAGO_GROUPS['arm'])
    # torso_joints = joints_from_names(tiago, TIAGO_GROUPS['torso'])
    # torso_left = torso_joints + left_joints
    print(get_link_pose(tiago, link_from_name(tiago, TIAGO_TOOL_FRAME)))
    # print(forward_kinematics('left', get_joint_positions(pr2, torso_left)))
    print(get_tool_pose(tiago))

    pose = get_tool_pose(tiago)
    generator = get_ik_generator(tiago, pose, torso_limits=False)
    for i in range(100):
        solutions = next(generator)
        print(i, len(solutions))
        for q in solutions:
            set_joint_positions(tiago, left_joints, q)
            if SLEEP is None:
                wait_if_gui('Continue?')
            else:
                wait_for_duration(SLEEP)

#####################################

def main():
    connect(use_gui=True)
    add_data_path()

    plane = create_floor()
    table = create_table()
    set_quat(table, quat_from_euler(Euler(yaw=PI/2)))
    obstacles = [table]
    tiago = create_tiago()

    base_start = (-2, -2, 0)
    base_goal = (2, 2, 0)
    arm_start = TUCKED_ARM
    arm_goal = EXTENDED_ARM

    arm_joints = joints_from_names(tiago, TIAGO_GROUPS['arm'])
    torso_joints = joints_from_names(tiago, TIAGO_GROUPS['torso'])
    set_joint_positions(tiago, arm_joints, arm_start)
    set_group_conf(tiago, 'torso', [0.33])
    open_gripper(tiago)
    set_base_values(tiago, base_start)
    
    # test inverse kinematics
    test_ikfast(tiago)

    # test base motion
    add_line(base_start, base_goal, color=RED)
    print(base_start, base_goal)
    test_base_motion(tiago, base_start, base_goal, obstacles=obstacles)
    base_goal = (4, 4, 0) # goal is w.r.t. start conf
    test_base_control(tiago, base_start, base_goal, obstacles=obstacles)

    # test arm motion
    test_arm_motion(tiago, arm_joints, arm_goal)
    test_arm_control(tiago, arm_joints, arm_start)

    wait_if_gui('Finish?')
    disconnect()

if __name__ == '__main__':
    main()