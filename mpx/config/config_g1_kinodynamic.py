import os
from functools import partial

import jax
import jax.numpy as jnp

import mpx.utils.models as mpc_dyn_model
import mpx.utils.mpc_wrapper as base_mpc_wrapper
import mpx.utils.objectives as mpc_objectives

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/unitree_g1/g1.xml"

# Named foot corner geoms in bundled g1.xml (see left_ankle_roll_link / right_ankle_roll_link).
# For MJCF without geom names, set `contact_sphere_on_bodies` in config instead (handled in MPCWrapper).
contact_frame = ["0_left", "0_right"]
body_name = ["left_ankle_roll_link", "right_ankle_roll_link"]

dt = 0.02
N = 25
mpc_frequency = 50

timer_t = jnp.array([0.5, 0.5])
duty_factor = 0.7
step_freq = 1.2
step_height = 0.08

initial_height = 0.755
robot_height = 0.755

p0 = jnp.array([0.0, 0.0, 0.755])
quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
q0 = jnp.zeros(23)

p_legs0 = jnp.zeros(6)

n_joints = 23
n_contact = len(contact_frame)
n = 13 + 2 * n_joints + 3 * n_contact
m = n_joints + 3 * n_contact
grf_as_state = False
u_ref = jnp.zeros(m)

Qp = jnp.diag(jnp.array([0.0, 0.0, 1e4]))
Qrot = jnp.diag(jnp.array([1.0, 1.0, 0.0])) * 1e3
Qq = jnp.diag(
    jnp.concatenate(
        [
            jnp.ones(12) * 4e0,
            jnp.ones(1) * 4e1,
            jnp.ones(10) * 4e1,
        ]
    )
)
Qdp = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e3
Qomega = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e0
Qleg = jnp.diag(jnp.tile(jnp.array([1e5, 1e5, 1e5]), n_contact))
Qdq_cmd = jnp.diag(jnp.ones(n_joints)) * 1e-2
Qgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qdq_cmd, Qgrf)

use_terrain_estimation = False
initial_state = jnp.concatenate([p0, quat0, q0, jnp.zeros(6 + n_joints), p_legs0])

joint_kp = jnp.array(
    [200.0, 200.0, 200.0, 200.0, 200.0, 60.0] * 2
    + [120.0]
    + [45.0, 45.0, 45.0, 45.0, 45.0]
    + [45.0, 45.0, 45.0, 45.0, 45.0]
)
joint_kd = jnp.array(
    [5.0, 5.0, 5.0, 5.0, 5.0, 1.5] * 2
    + [3.0]
    + [1.5, 1.5, 1.5, 1.5, 1.5]
    + [1.5, 1.5, 1.5, 1.5, 1.5]
)

torque_limits = jnp.array(
    [88.0, 88.0, 88.0, 139.0, 40.0, 40.0] * 2
    + [88.0]
    + [20.0, 20.0, 20.0, 20.0, 20.0]
    + [20.0, 20.0, 20.0, 20.0, 20.0]
)

cost = partial(mpc_objectives.g1_kinodynamic_obj, n_joints, n_contact, N)
hessian_approx = None


def dynamics(model, mjx_model, contact_id, body_id):
    return partial(
        mpc_dyn_model.g1_kinodynamic_dynamics,
        model,
        mjx_model,
        contact_id,
        body_id,
        n_joints,
        n_contact,
        dt,
    )


class MPCWrapper(base_mpc_wrapper.MPCWrapper):
    def __init__(self, config, limited_memory=False):
        super().__init__(config, limited_memory=limited_memory)
        self._torque_output = jax.jit(
            partial(
                mpc_dyn_model.g1_kinodynamic_torques,
                self.model,
                self.mjx_model,
                self.contact_id,
                self.body_id,
                n_joints,
                n_contact,
                dt,
                joint_kp,
                joint_kd,
            )
        )

    def control_output(self, x0, X, U, reference, parameter):
        return self._torque_output(x0, X, U, reference, parameter)


solver_mode = "primal_dual"

max_torque = torque_limits
min_torque = -torque_limits
