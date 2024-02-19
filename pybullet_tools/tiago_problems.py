from .pr2_problems import create_kitchen
from .pr2_utils import set_group_conf
from .tiago_utils import TIAGO_URDF, create_gripper, get_carry_conf, get_group_conf, open_gripper, set_arm_conf
from .utils import STATIC_MASS, HideOutput, LockRenderer, Point, Pose, create_body, create_shape_array, get_bodies, get_box_geometry, load_model, remove_body


class Problem(object):
    def __init__(self, robot, arms=tuple(), movable=tuple(), grasp_types=tuple(),
                 surfaces=tuple(), sinks=tuple(), stoves=tuple(), tools=tuple(), buttons=tuple(),
                 goal_conf=None, goal_holding=tuple(), goal_on=tuple(),
                 goal_cleaned=tuple(), goal_cooked=tuple(), costs=False,
                 body_names={}, body_types=[], base_limits=None):
        self.robot = robot
        self.movable = movable
        self.grasp_types = grasp_types
        self.surfaces = surfaces
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
        all_movable = [self.robot] + list(self.movable)
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
    
    
def create_tiago(fixed_base=True, torso=0.2):
    with LockRenderer():
        with HideOutput():
            tiago = load_model(TIAGO_URDF, fixed_base=fixed_base)
        set_group_conf(tiago, 'torso', [torso])
    return tiago

def create_hook(width=0.2, length=0.6, height=0.07, thickness=0.05, color=None, mass=STATIC_MASS):
    handle = get_box_geometry(length, thickness, height)
    handle_pose = Pose()

    head = get_box_geometry(thickness, width, height)
    head_pose = Pose(point=[length/2. - thickness/2., 0, 0])

    geoms = [handle] + [head]
    poses = [handle_pose] + [head_pose]
    colors = len(poses)*[color]

    collision_id, visual_id = create_shape_array(geoms, poses, colors)
    body = create_body(collision_id, visual_id, mass=mass)

    return body