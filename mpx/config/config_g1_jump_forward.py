from functools import partial
import mpx.config.config_g1_kinodynamic as base
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.objectives as mpc_objectives

task_name = "g1_jump_forward"

model_path = base.model_path
contact_frame = base.contact_frame
body_name = base.body_name

dt = 0.02
N = 50
mpc_frequency = base.mpc_frequency

timer_t = base.timer_t
duty_factor = 1.0
step_freq = base.step_freq
step_height = base.step_height
initial_height = base.initial_height
robot_height = base.robot_height

p0 = base.p0
quat0 = base.quat0
q0 = base.q0
p_legs0 = base.p_legs0

n_joints = base.n_joints
n_contact = base.n_contact
n = base.n
m = base.m
grf_as_state = base.grf_as_state
u_ref = base.u_ref
W = base.W

use_terrain_estimation = False
initial_state = base.initial_state

joint_kp = base.joint_kp
joint_kd = base.joint_kd
torque_limits = base.torque_limits

cost = partial(mpc_objectives.g1_kinodynamic_obj, n_joints, n_contact, N)
hessian_approx = base.hessian_approx
dynamics = base.dynamics
MPCWrapper = base.MPCWrapper

_crouch_leg_l = (-0.9, 1.6, -0.7)
_crouch_leg_r = (-0.9, 1.6, -0.7)
_crouch_arm_l = (0.3, 0.15, 0.0, 0.50)
_crouch_arm_r = (0.3, -0.15, 0.0, 0.50)

reference = partial(
    mpc_utils.reference_humanoid_jump_forward,
    base_height=robot_height,
    crouch_height=0.55,
    apex_height=0.95,
    jump_distance=0.6,
    foot_shift=0.32,
    foot_lift=0.08,
    crouch_left=(0, 3, 4, 13, 14, 15, 16),
    crouch_left_vals = _crouch_leg_l + _crouch_arm_l,
    crouch_right=(6, 9, 10, 18, 19, 20, 21),
    crouch_right_vals= _crouch_leg_r + _crouch_arm_r,
)

solver_mode = "fddp"
max_torque = base.max_torque
min_torque = base.min_torque