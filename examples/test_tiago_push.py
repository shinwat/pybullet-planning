#!/usr/bin/env python

from __future__ import print_function
import random
import argparse
from pybullet_tools.tiago_problems import Problem, create_tiago
from pybullet_tools.pr2_problems import TABLE_MAX_Z, create_floor, create_table
from pybullet_tools.tiago_utils import EXTENDED_ARM, TIAGO_GROUPS, TIAGO_URDF, TUCKED_ARM, get_carry_conf, get_gripper_joints, get_group_conf, get_group_joints, open_gripper, set_arm_conf, set_group_conf

from pybullet_tools.pr2_primitives import Pose, Conf, Command, State
from pybullet_tools.tiago_primitives import Attach, Detach, GripperCommand, Push, get_align_gen, get_grasp_gen, get_ik_fn, get_ik_ir_gen, get_ik_ir_traj_gen, get_ik_traj_fn, get_motion_gen, get_push_gen, get_stable_gen, control_commands, apply_commands
from pybullet_tools.kuka_primitives import BodyConf, Command, get_free_motion_gen, get_holding_motion_gen
from pybullet_tools.utils import RED, WorldSaver, clone_body, create_box, create_sphere, enable_gravity, connect, dump_world, get_aabb, get_max_limit, get_pose, joints_from_names, multiply, placement_on_aabb, point_from_pose, pose_from_base_values, pose_from_pose2d, sample_placement, set_base_values, set_joint_positions, set_point, set_pose, \
    draw_global_system, draw_pose, set_camera_pose, Point, set_default_camera, stable_z, \
    BLOCK_URDF, load_model, wait_if_gui, disconnect, DRAKE_IIWA_URDF, wait_if_gui, update_state, disable_real_time, HideOutput
from pybullet_tools.utils import Pose as Posee

def plan(problem, max_attempts, collisions, directory, policy, slide, move_base):
    robot = problem.robot
    block = problem.movable[0]
    table = problem.surfaces[0]

    # define generators (streams)
    align_gen_fn = get_align_gen(problem, collisions=collisions)
    ik_ir_fn = get_ik_ir_traj_gen(problem, max_attempts=max_attempts, collisions=collisions)
    ik_fn = get_ik_traj_fn(problem, collisions=collisions)
    motion_gen_fn = get_motion_gen(problem, teleport=False)
    push_gen_fn = get_push_gen(problem, collisions=collisions)

    # find align pose
    body_pose = placement_on_aabb(block, get_aabb(table))
    pose0 = Pose(block, body_pose, table) # current block pose
    pose1 = Pose(block, Posee(point=point_from_pose(body_pose)+slide), table) # target block pose
    print(pose1.value)
    pose_start = Pose(problem.robot, get_pose(problem.robot)) # recover robot pose from start
    close_gripper = GripperCommand(problem.robot, 0.0)
    gripper_joint = get_gripper_joints(problem.robot)[0]
    position = get_max_limit(problem.robot, gripper_joint)
    open_gripper = GripperCommand(problem.robot, position)
    saved_world = WorldSaver()
    for _ in range(max_attempts):
        (g,) = list(align_gen_fn(block, pose0, pose1))
        if move_base:
            output = next(ik_ir_fn('', block, pose0, pose1, g), None)
            if output is None:
                print('Failed to find a solution')
            else:
                (bq, aq, ac1) = output # base config + arm config + arm trajectory commands
                #  plan base
                saved_world.restore() # restore world before planning base motion
                pose_goal = Pose(problem.robot, pose_from_pose2d(bq.values))
                goal_conf = pose_goal.to_base_conf()
                (bc,) = motion_gen_fn(pose_start, goal_conf) #x,y,yaw
                [bt,] = bc.commands # base trajectory
                [at1,] = ac1.commands # arm trajectory grasp
                print(bq.values)
                pose2 = Pose(block, multiply(pose1.value, g.value))
                output2 = push_gen_fn('', block, pose0, pose2, g, goal_conf, aq)
                if output2 is None:
                    print("Failed to find push path.")
                else:
                    (ac2,) = output2
                    [at2,] = ac2.commands # arm trajectory push
                    push = Push(robot, block, pose1, at2, directory, policy)
                    return (bt,close_gripper,at1,push,push.reverse(),open_gripper)
        else: # static base
            goal_conf = pose_start.to_base_conf()
            output = ik_fn('', block, pose0, pose1, g, goal_conf)
            if output is None:
                print('Failed to find a solution')
            else:
                (aq, ac1) = output
                saved_world.restore() # restore world before planning base motion
                [at1,] = ac1.commands
                output2 = push_gen_fn('', block, pose0, pose1, g, goal_conf, aq)
                if output2 is None:
                    print("Failed to find push path.")
                else:
                    (ac2,) = output2
                    [at2,] = ac2.commands # arm trajectory push
                    push = Push(robot, block, pose1, at2, directory, policy)
                    return (bt,close_gripper,at1,push,push.reverse(),open_gripper)
    return None

def pushing(grasp_type='top'):
    initial_conf = get_carry_conf(grasp_type)
    tiago = create_tiago()
    table = create_table(height=TABLE_MAX_Z)
    floor = create_floor()
    block = create_box(.07, .07, .14, mass=0.05)
    set_point(block, (0, 0, TABLE_MAX_Z + .15/2))
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    parser.add_argument('-cfree', action='store_true', help='Disables collisions')
    parser.add_argument('-d','--directory', type=str, default=None, help='path to save trajectory')
    parser.add_argument('-p','--policy', type=str, default=None, help='path to the policy if available')
    parser.add_argument('-static', action='store_true', help='base is static')
    parser.add_argument('-direct', action='store_true', help='no GUI')
    parser.add_argument('-random', action='store_true', help='random goal placement')
    parser.add_argument('-x','--x', type=float, default=0.0, help='Goal displacement in x')
    parser.add_argument('-y','--y', type=float, default=0.0, help='Goal displacement in y')
    args = parser.parse_args()
    print('Arguments:', args)
    
    connect(use_gui=not args.direct)
    disable_real_time()
    draw_global_system()
    with HideOutput():
        problem = pushing()
    set_default_camera(distance=2)
    dump_world()

    if args.random:
        x, y = (random.uniform(-0.27, 0.27), random.uniform(-0.27, 0.27))
    else:
        x, y = (args.x, args.y)
    goal = (x, y, TABLE_MAX_Z)
    set_point(create_sphere(radius=0.02, mass=0.01, color=RED, collision=False), goal) # visualize goal in simulator
    saved_world = WorldSaver()
    commands = plan(problem, 25, not args.cfree, args.directory, args.policy, Point(x=x,y=y), move_base=not args.static) # 500 attempts
    
    if (commands is None) or (display is None):
        print('Unable to find a plan!')
        return

    saved_world.restore()

    update_state()
    wait_if_gui('{}?'.format(display))

    # commands = post_process(problem, plan)
    if args.simulate:
        control_commands(commands)
    else:
        apply_commands(State(), commands, time_step=0.05)

    print('Quit?')
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()