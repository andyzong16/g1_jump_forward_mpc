from functools import partial
import mpx.config.config_g1_kinodynamic as base
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.objectives as mpc_objectives

task_name = "g1_jump_forward"

model_path = base.model_path
contact_frame = [
    "0_left",
    "0_right",
]
body_name = ["left_ankle_roll_link", "right_ankle_roll_link"]

dt = 0.02
# Keep horizon moderate for 29-DoF MJX compile/runtime.
N = 60
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
n=base.n
m=base.m
grf_as_state = base.grf_as_state
u_ref = base.u_ref
W= base.W
W = W.at[3, 3].set(3.0e3)   # base orientation
W = W.at[4, 4].set(5.0e3)   # pitch orientation

for idx in (18, 19, 20):    # waist joints in q block
    W = W.at[idx, idx].set(3.0e2)
W = W.at[20, 20].set(7.5e2)  # waist pitch: discourage leaning back
for idx in (53, 54, 55):    # waist joint velocity block
    W = W.at[idx, idx].set(8.0e1)
W = W.at[55, 55].set(1.5e2)  # damp waist pitch during takeoff/landing
for idx in (88, 89, 90):  # waist command regularization block
    W = W.at[idx, idx].set(1.0e1)
W = W.at[90, 90].set(2.0e1)  # smoother waist pitch commands for training

use_terrain_estimation = False
initial_state = base.initial_state

joint_kp = base.joint_kp
joint_kd = base.joint_kd
torque_limits = base.torque_limits

cost = partial(mpc_objectives.g1_jump_forward_obj, n_joints, n_contact, N)
hessian_approx = base.hessian_approx
dynamics = base.dynamics
MPCWrapper = base.MPCWrapper

_crouch_leg_l = (-0.9, 0.0, 0.0, 1.6, -0.7, 0.0)
_crouch_leg_r = (-0.9, 0.0, 0.0, 1.6, -0.7, 0.0)
_crouch_arm_l = (0.3, 0.15, 0.0, 0.50)
_crouch_arm_r = (0.3, -0.15, 0.0, 0.50)

reference = partial(
    mpc_utils.reference_humanoid_jump_forward,
    base_height=robot_height,
    crouch_height=0.68,
    apex_height=0.82,
    jump_distance=0.95,
    foot_shift=0.50,
    foot_lift=0.05,
    crouch_left=(0, 1, 2, 3, 4, 5, 15, 16, 17, 18),
    crouch_left_vals=_crouch_leg_l + _crouch_arm_l,
    crouch_right=(6, 7, 8, 9, 10, 11, 22, 23, 24, 25),
    crouch_right_vals=_crouch_leg_r + _crouch_arm_r,
)

solver_mode = "primal_dual"
max_torque = base.max_torque
min_torque = base.min_torque