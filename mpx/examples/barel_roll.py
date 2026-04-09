import argparse
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mujoco
import numpy as np

from gym_quadruped.quadruped_env import QuadrupedEnv

import mpx.config.config_barrel_roll as config
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.utils.sim as sim_utils

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def main(headless=False):
    env = QuadrupedEnv(
        robot="aliengo",
        scene="flat",
        sim_dt=1 / 200.0,
        ref_base_lin_vel=0.0,
        ground_friction_coeff=1.5,
        base_vel_command_type="human",
        state_obs_names=tuple(QuadrupedEnv.ALL_OBS),
    )
    env.reset(random=False)

    mpc = mpc_wrapper.MPCWrapper(config, limited_memory=True)
    qpos0 = jnp.concatenate([config.p0, config.quat0, config.q0])
    qvel0 = jnp.zeros(6 + config.n_joints)

    X, U, reference, output = mpc.runOffline(qpos0, qvel0)
    if headless:
        print("Offline solve shapes:", X.shape, U.shape, reference.shape, len(output))
        return

    env.mjData.qpos = qpos0
    env.render()
    ids = []
    for _ in range(config.n_contact):
        ids.append(
            sim_utils.render_vector(
                env.viewer,
                np.zeros(3),
                np.zeros(3),
                0.1,
                np.array([1, 0, 0, 1]),
            )
        )

    counter = 0
    iteration = 99
    ghost_geoms = None
    ghost_scratch = mujoco.MjData(env.mjModel)
    while env.viewer.is_running():
        env.mjData.qpos = X[counter, : 7 + config.n_joints]
        env.mjData.qvel = X[counter, 7 + config.n_joints : 13 + 2 * config.n_joints]

        if iteration < len(output):
            ghost_data = output[iteration][::10, : 7 + config.n_joints]
            ghost_geoms, ghost_scratch = sim_utils.render_ghost_trajectory(
                env.viewer,
                env.mjModel,
                ghost_data,
                np.arange(ghost_data.shape[0]) / ghost_data.shape[0] * 0.5,
                ghost_geoms=ghost_geoms,
                scratch_data=ghost_scratch,
            )
        else:
            ghost_data = output[-1][::10, : 7 + config.n_joints]
            ghost_geoms, ghost_scratch = sim_utils.render_ghost_trajectory(
                env.viewer,
                env.mjModel,
                ghost_data,
                np.arange(ghost_data.shape[0]) * 0.0,
                ghost_geoms=ghost_geoms,
                scratch_data=ghost_scratch,
            )
        iteration += 1

        grf = X[
            counter,
            13
            + 2 * config.n_joints
            + 3 * config.n_contact : 13
            + 2 * config.n_joints
            + 6 * config.n_contact,
        ]
        foot = X[
            counter,
            13 + 2 * config.n_joints : 13 + 2 * config.n_joints + 3 * config.n_contact,
        ]
        for idx in range(config.n_contact):
            tangential_force = np.sqrt(grf[3 * idx] ** 2 + grf[3 * idx + 1] ** 2)
            color = (
                np.array([1, 0, 0, 1])
                if grf[3 * idx + 2] * 0.5 < tangential_force
                else np.array([0, 1, 0, 1])
            )
            sim_utils.render_vector(
                env.viewer,
                grf[3 * idx : 3 * (idx + 1)],
                foot[3 * idx : 3 * (idx + 1)],
                np.linalg.norm(grf[3 * idx : 3 * (idx + 1)]) / 220,
                color,
                ids[idx],
            )

        mujoco.mj_step(env.mjModel, env.mjData)
        if iteration < len(output):
            time.sleep(config.dt)
        else:
            counter += 1
            time.sleep(5 * config.dt)
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(headless=args.headless)
