#!/usr/bin/env python

from __future__ import print_function

from pybullet_tools.ikfast.tiago.ik import sample_tool_ik
from pybullet_tools.tiago_problems import create_tiago
from pybullet_tools.pr2_problems import create_floor
from pybullet_tools.tiago_utils import TUCKED_ARM, TIAGO_GROUPS, get_arm_joints, open_gripper, set_group_conf
from pybullet_tools.utils import Point, Pose, invert, joint_controller_hold, multiply, set_joint_positions, \
    add_data_path, connect, joints_from_names, step_simulation, wait_if_gui, disconnect, \
    wait_if_gui, Euler, wait_for_duration

SLEEP = 0.01

def test_ikfast(tiago):
    wait_if_gui('Test IK?')
    from pybullet_tools.ikfast.tiago.ik import get_tool_pose, get_ik_generator

    # TEST 1: checking how often IK fails with gripper in place
    arm_joints = get_arm_joints(tiago)    
    gripper_pose = get_tool_pose(tiago)
    generator = get_ik_generator(tiago, gripper_pose, torso_limits=False)
    failures = 0
    for i in range(100):
        solutions = next(generator)
        if len(solutions) == 0:
            failures += 1
        for q in solutions:
            set_joint_positions(tiago, arm_joints, q)
            if SLEEP is None:
                wait_if_gui('Continue?')
            else:
                wait_for_duration(SLEEP)
    print('# failures: ', failures)

    wait_if_gui('Continue?')

    # TEST 2: checking IK with moving gripper
    pose_change = Pose(Point(0.0, -0.0, -0.01), Euler(0,0,0))
    for _ in range(100):
        gripper_pose = multiply(pose_change, invert(gripper_pose))
        torso_arm_conf = sample_tool_ik(tiago, gripper_pose, max_attempts=100)
        print(torso_arm_conf) #KLUDGE: every other time it's None
        for i, _ in enumerate(joint_controller_hold(tiago, arm_joints, torso_arm_conf)):
            step_simulation()
                
def main():
    connect(use_gui=True)
    add_data_path()

    create_floor()
    tiago = create_tiago()
    arm_start = TUCKED_ARM
    arm_joints = joints_from_names(tiago, TIAGO_GROUPS['arm'])
    set_joint_positions(tiago, arm_joints, arm_start)
    set_group_conf(tiago, 'torso', [0.33])
    open_gripper(tiago)
    
    # test inverse kinematics
    test_ikfast(tiago)

    wait_if_gui('Finish?')
    disconnect()

if __name__ == '__main__':
    main()