#!/usr/bin/env python

from __future__ import print_function

import pybullet as p
import time

from pybullet_tools.tiago_utils import TIAGO_URDF, TUCKED_ARM, EXTENDED_ARM, TIAGO_GROUPS, open_gripper
from pybullet_tools.utils import set_base_values, joint_from_name, quat_from_euler, set_joint_position, \
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
        #wait_if_gui('Continue?')
        wait_for_duration(0.01)

#####################################

def main():
    connect(use_gui=True)
    add_data_path()

    plane = p.loadURDF("plane.urdf")
    table_path = "models/table_collision/table.urdf"
    table = load_pybullet(table_path, fixed_base=True)
    set_quat(table, quat_from_euler(Euler(yaw=PI/2)))
    obstacles = [table] # TODO: include plane in obstacles

    tiago_urdf = TIAGO_URDF
    with HideOutput():
        tiago = load_model(tiago_urdf, fixed_base=True) # TODO: suppress warnings?
    dump_body(tiago)

    base_start = (-2, -2, 0)
    base_goal = (2, 2, 0)
    arm_start = TUCKED_ARM
    arm_goal = EXTENDED_ARM

    arm_joints = joints_from_names(tiago, TIAGO_GROUPS['arm'])
    torso_joints = joints_from_names(tiago, TIAGO_GROUPS['torso'])
    set_joint_positions(tiago, arm_joints, arm_start)
    set_joint_positions(tiago, torso_joints, [0.2])
    open_gripper(tiago)
    set_base_values(tiago, base_start)

    # test base motion
    add_line(base_start, base_goal, color=RED)
    print(base_start, base_goal)
    test_base_motion(tiago, base_start, base_goal, obstacles=obstacles)

    # test arm motion
    test_arm_motion(tiago, arm_joints, arm_goal)

    wait_if_gui('Finish?')
    disconnect()
if __name__ == '__main__':
    main()