import time
import numpy as np
from .tiago_utils import TIAGO_URDF, create_gripper, set_group_conf
from .utils import (
    GREEN,
    STATIC_MASS,
    TAN,
    HideOutput,
    LockRenderer,
    Point,
    Pose,
    create_body,
    create_collision_shape,
    create_shape_array,
    create_visual_shape,
    get_bodies,
    get_box_geometry,
    get_cylinder_geometry,
    load_model,
    remove_body,
)

LIGHT_GREY = (0.7, 0.7, 0.7, 1.0)


class Problem(object):
    def __init__(
        self,
        robot,
        arms=tuple(),
        movable=tuple(),
        grasp_types=tuple(),
        surfaces=tuple(),
        bumps=tuple(),
        sinks=tuple(),
        stoves=tuple(),
        tools=tuple(),
        buttons=tuple(),
        goal_conf=None,
        goal_holding=tuple(),
        goal_on=tuple(),
        goal_cleaned=tuple(),
        goal_cooked=tuple(),
        costs=False,
        body_names={},
        body_types=[],
        base_limits=None,
    ):
        self.robot = robot
        self.movable = movable
        self.grasp_types = grasp_types
        self.surfaces = surfaces
        self.bumps = bumps
        self.sinks = sinks
        self.stoves = stoves
        self.tools = tools
        self.buttons = buttons
        self.goal_conf = goal_conf
        self.goal_holding = goal_holding
        self.goal_on = goal_on
        self.goal_cleaned = goal_cleaned
        self.goal_cooked = goal_cooked
        self.costs = costs
        self.body_names = body_names
        self.body_types = body_types
        self.base_limits = base_limits
        all_movable = [self.robot] + list(self.movable) + list(self.bumps)
        self.fixed = list(filter(lambda b: b not in all_movable, get_bodies()))
        self.gripper = None

    def get_gripper(self, visual=True):
        if self.gripper is None:
            self.gripper = create_gripper(self.robot, visual=visual)
        return self.gripper

    def remove_gripper(self):
        if self.gripper is not None:
            remove_body(self.gripper)
            self.gripper = None

    def __repr__(self):
        return repr(self.__dict__)


def create_tiago(fixed_base=True, torso=0.2, max_retries=5, delay_seconds=1):
    with LockRenderer():
        with HideOutput():
            for attempt in range(1, max_retries+1):
                try:
                    tiago = load_model(TIAGO_URDF, fixed_base=fixed_base)
                    set_group_conf(tiago, "torso", [torso])
                    return tiago
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    print(f"Attempt {attempt} failed: {e}. "
                        f"Retrying in {delay_seconds:.3f}s...")
                    time.sleep(delay_seconds)


def create_hook(
    width=0.2, length=0.6, height=0.07, thickness=0.05, color=None, mass=STATIC_MASS
):
    handle = get_box_geometry(length, thickness, height)
    handle_pose = Pose()

    head = get_box_geometry(thickness, width, height)
    head_pose = Pose(point=[length / 2.0 - thickness / 2.0, 0, 0])

    geoms = [handle] + [head]
    poses = [handle_pose] + [head_pose]
    colors = len(poses) * [color]

    collision_id, visual_id = create_shape_array(geoms, poses, colors)
    body = create_body(collision_id, visual_id, mass=mass)

    return body


# table with one leg in the center
def create_table(
    width=0.6,
    length=1.2,
    height=0.73,
    thickness=0.03,
    radius=0.015,
    top_color=LIGHT_GREY,
    leg_color=TAN,
    cylinder=True,
    **kwargs,
):
    surface = get_box_geometry(width, length, thickness)
    surface_pose = Pose(Point(z=height - thickness / 2.0))

    leg_height = height - thickness
    if cylinder:
        leg_geometry = get_cylinder_geometry(radius, leg_height)
    else:
        leg_geometry = get_box_geometry(
            width=2 * radius, length=2 * radius, height=leg_height
        )
    legs = [leg_geometry for _ in range(1)]
    leg_poses = [Pose(point=[x, y, leg_height / 2.0]) for x, y in [np.array((0, 0))]]

    geoms = [surface] + legs
    poses = [surface_pose] + leg_poses
    colors = [top_color] + len(legs) * [leg_color]

    collision_id, visual_id = create_shape_array(geoms, poses, colors)
    body = create_body(collision_id, visual_id, **kwargs)

    return body


def create_plate(r, h, mass=STATIC_MASS, color=GREEN, **kwargs):
    collision_id = create_collision_shape(get_cylinder_geometry(r, 0.0))
    visual_id = create_visual_shape(get_cylinder_geometry(r, h), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)
