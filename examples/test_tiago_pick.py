#!/usr/bin/env python

from __future__ import print_function
import argparse
import random
import time
from pybullet_tools.tiago_problems import Problem, create_tiago
from pybullet_tools.pr2_problems import TABLE_MAX_Z, create_floor, create_table
from pybullet_tools.tiago_utils import EXTENDED_ARM, TIAGO_GROUPS, TIAGO_URDF, TUCKED_ARM, get_carry_conf, get_group_conf, get_group_joints, open_gripper, set_arm_conf, set_group_conf

from pybullet_tools.pr2_primitives import Pose, State
from pybullet_tools.tiago_primitives import Conf, apply_commands, control_commands, get_grasp_gen, get_ik_fn, get_ik_ir_gen, get_motion_gen, get_stable_gen
from pybullet_tools.kuka_primitives import BodyConf, Command, get_free_motion_gen, get_holding_motion_gen
from pybullet_tools.utils import WorldSaver, clone_body, create_box, enable_gravity, connect, dump_world, enable_real_time, get_aabb, get_pose, joint_controller_hold, joints_from_names, placement_on_aabb, pose_from_base_values, pose_from_pose2d, sample_placement, set_base_values, set_joint_positions, set_point, set_pose, \
    draw_global_system, draw_pose, set_camera_pose, Point, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, step_simulation, wait_if_gui, disconnect, DRAKE_IIWA_URDF, wait_if_gui, update_state, disable_real_time, HideOutput


def plan(problem, max_attempts, move_base):
    robot = problem.robot
    block = problem.movable[0]
    table = problem.surfaces[0]

    # define generators (streams)
    grasp_gen_fn = get_grasp_gen(problem, collisions=True)
    ik_ir_fn = get_ik_ir_gen(problem, max_attempts=max_attempts, collisions=True)
    ik_fn = get_ik_fn(problem, collisions=True)
    motion_gen_fn = get_motion_gen(problem, teleport=False)

    # find grasp pose
    grasps = list(grasp_gen_fn(block))
    print('Grasps: ', len(grasps))
    body_pose = placement_on_aabb(block, get_aabb(table))
    pose0 = Pose(block, body_pose, table)
    pose_start = get_pose(robot)
    conf0 = Conf(robot, get_group_joints(robot, 'base'), get_group_conf(robot, 'base'))
    saved_world = WorldSaver()
    (g,) = random.choice(grasps)

    for _ in range(max_attempts):
        if move_base:
            output = next(ik_ir_fn('', block, pose0, g), None)
            if output is None:
                print('Failed to find a solution')
            else:
                (bq, ac) = output # base pose + arm commands
                #  plan base
                saved_world.restore() # restore world before planning base motion
                pose_start = Pose(problem.robot, get_pose(problem.robot)) # recover robot pose from start
                pose_goal = Pose(problem.robot, pose_from_pose2d(bq.values))
                goal_conf = pose_goal.to_base_conf()
                (bc,) = motion_gen_fn(pose_start, goal_conf) #x,y,yaw
                [bt,] = bc.commands # base trajectory
                path1 = bt
                [at,] = ac.commands # arm trajectory
                path2 = at
                return (path1,path2,)
        else:
            result1 = ik_fn('', block, pose0, g, conf0)
            if result1 is None:
                print('Failed to find a solution')
            else:
                (ac,) = result1
                [path2,] = ac.commands
                return (path2,)
    return None

def picking(grasp_type='top'):
    initial_conf = get_carry_conf(grasp_type)
    tiago = create_tiago()
    table = create_table(height=TABLE_MAX_Z)
    floor = create_floor()
    block = create_box(.07, .07, .14)
    set_point(block, (0, 0, TABLE_MAX_Z + .15)) #/2
    base_start = (-0.7, 0, 0)

    # set initial robot config
    set_group_conf(tiago, 'torso', [0.33])
    set_arm_conf(tiago, initial_conf)
    open_gripper(tiago)
    set_base_values(tiago, base_start)

    return Problem(robot=tiago, movable=[block], arms=[], grasp_types=[grasp_type],
                   surfaces=[table], sinks=[], stoves=[],
                   goal_cooked=[])

def main(display='execute'): # control | execute | step
    parser = argparse.ArgumentParser()  # Automatically includes help
    parser.add_argument('-static', action='store_true', help='keep base static.')
    parser.add_argument('-simulate', action='store_true', help='enable viewer.')
    args = parser.parse_args()
    connect(use_gui=True)
    disable_real_time()
    draw_global_system()
    with HideOutput():
        problem = picking()
    set_default_camera(distance=2)
    dump_world()

    saved_world = WorldSaver()
    wait_if_gui()
    commands = plan(problem, 2500, move_base=not args.static)
    
    if (commands is None) or (display is None):
        print('Unable to find a plan!')
        return
    print(commands)
    saved_world.restore()

    update_state()
    wait_if_gui('{}?'.format(display))
    if args.simulate:
    # control_commands(commands)
    # enable_gravity()
    # enable_real_time()
    # for i in range(100):
    #     step_simulation() #TODO: remove
    #     wait_if_gui()
        for command in commands:
            for conf in command.path:
                for _ in joint_controller_hold(conf.body, conf.joints, conf.values):
                    step_simulation()
                    time.sleep(0)
    else:
        apply_commands(State(), commands, time_step=0.01)
    print('Quit?')
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()