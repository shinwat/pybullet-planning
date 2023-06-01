#!/usr/bin/env python

from __future__ import print_function
from pybullet_tools.pr2_utils import PR2_GROUPS, get_carry_conf, get_group_conf, get_group_joints, open_arm, set_arm_conf
from pybullet_tools.pr2_problems import Problem, create_floor, create_pr2, create_table

from pybullet_tools.pr2_primitives import Attach, GripperCommand, Pose, Conf, Command, State, apply_commands, get_grasp_gen, get_ik_fn, get_ik_ir_gen
from pybullet_tools.utils import WorldSaver, clone_body, enable_gravity, connect, dump_world, get_pose, joints_from_names, set_base_values, set_joint_positions, set_point, set_pose, \
    draw_global_system, draw_pose, set_camera_pose, Point, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_if_gui, disconnect, DRAKE_IIWA_URDF, wait_if_gui, update_state, disable_real_time, HideOutput


def plan(problem, teleport):
    robot = problem.robot
    block = problem.movable[0]
    fixed = problem.fixed

    # define generators (streams)
    grasp_gen = get_grasp_gen(problem, 'top')
    ik_fn = get_ik_fn(problem, teleport=teleport)

    pose0 = Pose(block, get_pose(block), init=True)
    conf0 = Conf(robot, get_group_joints(robot, 'base'), get_group_conf(robot, 'base'))
    saved_world = WorldSaver()
    grasps = grasp_gen(block)
    print('Grasps: ', len(grasps))
    for grasp, in grasps:
        saved_world.restore()
        print('width: ', grasp.grasp_width)
        print('value: ', grasp.value)
        result1 = ik_fn(problem.arms[0], block, pose0, grasp, conf0)
        if result1 is None:
            continue
        (ac,) = result1
        [path2,] = ac.commands
        pose0.assign()
        return (path2,)
    return None

def picking(arm='left', grasp_type='top'):
    initial_conf = get_carry_conf(arm, grasp_type)
    pr2 = create_pr2()
    table = create_table()
    floor = create_floor()
    block = load_model(BLOCK_URDF, fixed_base=False)
    base_start = (-0.5, 0, 0)

    # set initial robot config
    set_arm_conf(pr2, arm, initial_conf)
    open_arm(pr2, arm)
    set_base_values(pr2, base_start)
    set_point(block, (0, 0, stable_z(block, table)))
    return Problem(robot=pr2, movable=[block], arms=[arm], grasp_types=[grasp_type],
                   surfaces=[table], sinks=[], stoves=[],
                   goal_cooked=[])

def main(display='execute'):
    connect(use_gui=True)
    disable_real_time()
    draw_global_system()
    with HideOutput():
        problem = picking()
    set_default_camera(distance=2)
    dump_world()

    saved_world = WorldSaver()
    wait_if_gui()
    commands = plan(problem, teleport=False)
    if (commands is None) or (display is None):
        print('Unable to find a plan!')
        return

    saved_world.restore()
    update_state()
    wait_if_gui('{}?'.format(display))
    apply_commands(State(), commands)

    print('Quit?')
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()