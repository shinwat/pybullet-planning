#!/usr/bin/env python

import argparse
import random
import time

from pybullet_tools.tiago_utils import TIAGO_URDF, create_gripper, get_base_pose, set_arm_conf, \
    get_carry_conf, get_gripper_link, set_group_conf
from pybullet_tools.utils import create_box, disconnect, add_data_path, connect, multiply, invert, \
    get_link_pose, load_pybullet, HideOutput, wait_if_gui, elapsed_time
from pybullet_tools.pr2_problems import create_table
from pybullet_tools.tiago_primitives import get_stable_gen, get_grasp_gen, get_ik_ir_gen

class MockProblem(object):
    def __init__(self, robot, fixed=[], grasp_types=[]):
        self.robot = robot
        self.fixed = fixed
        self.grasp_types = grasp_types
        self.gripper = None
    def get_gripper(self, visual=True):
        if self.gripper is None:
            self.gripper = create_gripper(self.robot, visual=True)
        return self.gripper

def create_inverse_reachability2(robot, body, table, grasp_type, max_attempts=500, num_samples=500):
    tool_link = get_gripper_link(robot)
    problem = MockProblem(robot, fixed=[table], grasp_types=[grasp_type])
    placement_gen_fn = get_stable_gen(problem)
    grasp_gen_fn = get_grasp_gen(problem, collisions=True)
    ik_ir_fn = get_ik_ir_gen(problem, max_attempts=max_attempts, learned=False, teleport=True)
    placement_gen = placement_gen_fn(body, table)
    grasps = list(grasp_gen_fn(body))
    print('Grasps:', len(grasps))

    # TODO: sample the torso height
    # TODO: consider IK with respect to the torso frame
    start_time = time.time()
    gripper_from_base_list = []
    while len(gripper_from_base_list) < num_samples:
        (p,) = next(placement_gen)
        (g,) = random.choice(grasps)
        output = next(ik_ir_fn('blah', body, p, g), None)
        if output is None:
            print('Failed to find a solution after {} attempts'.format(max_attempts))
        else:
            (_, ac) = output
            [at,] = ac.commands
            at.path[-1].assign()
            gripper_from_base = multiply(invert(get_link_pose(robot, tool_link)), get_base_pose(robot))
            gripper_from_base_list.append(gripper_from_base)
            print('{} / {} [{:.3f}]'.format(
                len(gripper_from_base_list), num_samples, elapsed_time(start_time)))
            wait_if_gui()
    return None #save_inverse_reachability(robot, arm, grasp_type, tool_link, gripper_from_base_list)

#######################################################

def main():
    parser = argparse.ArgumentParser()  # Automatically includes help
    parser.add_argument('-grasp', required=True)
    parser.add_argument('-viewer', action='store_true', help='enable viewer.')
    args = parser.parse_args()

    grasp_type = args.grasp

    connect(use_gui=args.viewer)
    add_data_path()

    with HideOutput():
        robot = load_pybullet(TIAGO_URDF)
    set_group_conf(robot, 'torso', [0.3])
    set_arm_conf(robot, get_carry_conf(grasp_type))

    table = create_table()
    box = create_box(.07, .07, .14)

    create_inverse_reachability2(robot, box, table, grasp_type=grasp_type)
    disconnect()

if __name__ == '__main__':
    main()