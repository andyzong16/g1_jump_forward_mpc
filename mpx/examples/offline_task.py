import argparse
import importlib
import os
import sys
import time
from types import SimpleNamespace

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.utils.mpc_wrapper as base_mpc_wrapper
import mpx.utils.offline_solver as offline_solver
import mpx.utils.sim as sim_utils

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


TASKS = {
    "barrel_roll": {
        "config": "mpx.config.config_barrel_roll",
        "scene_path": os.path.abspath(
            os.path.join(dir_path, "..", "data", "aliengo", "scene_flat.xml")
        ),
        "benchmark_mode": "wrapper",
    },
    "h1_jump_forward": {
        "config": "mpx.config.config_h1_jump_forward",
        "scene_path": os.path.abspath(
            os.path.join(dir_path, "..", "data", "unitree_h1", "mjx_scene_h1_walk.xml")
        ),
        "benchmark_mode": "wrapper",
    },
    "aliengo_trot_two_step": {
        "config": "mpx.config.config_aliengo_trot_two_step",
        "scene_path": os.path.abspath(
            os.path.join(dir_path, "..", "data", "aliengo", "scene_flat.xml")
        ),
        "benchmark_mode": "wrapper",
    },
    "acrobot_swingup": {
        "config": "mpx.config.config_acrobot_swingup",
        "benchmark_mode": "direct",
    },
    "g1_jump_forward": {
        "config": "mpx.config.config_g1_jump_forward",
        # scene.xml includes g1.xml plus skybox / checker floor (bare g1.xml has no backdrop).
        "scene_path": os.path.abspath(
            os.path.join(dir_path, "..", "data", "unitree_g1", "scene.xml")
        ),
        "benchmark_mode": "wrapper",
    }
}

SOLVERS = ("primal_dual", "fddp")

ISAAC_G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

ISAAC_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

MPX_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _clone_config(module_name, solver_mode):
    module = importlib.import_module(module_name)
    attrs = {name: getattr(module, name) for name in dir(module) if not name.startswith("__")}
    config = SimpleNamespace(**attrs)
    if solver_mode is not None:
        config.solver_mode = solver_mode
    return config


def _solve_wrapper_task(config, max_iter, verbose):
    wrapper_cls = getattr(config, "MPCWrapper", base_mpc_wrapper.MPCWrapper)
    mpc = wrapper_cls(config, limited_memory=True)
    qpos0 = getattr(config, "offline_qpos0", jnp.concatenate([config.p0, config.quat0, config.q0]))
    qvel0 = getattr(config, "offline_qvel0", jnp.zeros(6 + config.n_joints))
    X, U, reference, history, stats = mpc.runOffline(
        qpos0,
        qvel0,
        return_stats=True,
        verbose=verbose,
        max_iter=max_iter,
    )
    return {
        "config": config,
        "X": X,
        "U": U,
        "reference": reference,
        "history": history,
        "stats": stats,
        "initial_state": {"qpos0": qpos0, "qvel0": qvel0},
    }


def _solve_direct_task(config, max_iter, verbose):
    _, solve = base_mpc_wrapper.build_solver_step(
        config,
        config.cost,
        config.dynamics,
        config.hessian_approx,
        limited_memory=False,
    )
    solve = jax.jit(solve)
    X, U, V, history, stats = offline_solver.run_offline_solve(
        solve,
        config.cost,
        config.dynamics,
        config.solver_mode,
        config.reference,
        config.parameter,
        config.W,
        config.x0,
        config.initial_X0,
        config.initial_U0,
        config.initial_V0,
        max_iter=max_iter,
        verbose=verbose,
    )
    return {
        "config": config,
        "X": X,
        "U": U,
        "V": V,
        "reference": config.reference,
        "history": history,
        "stats": stats,
        "initial_state": {"x0": config.x0},
    }


def solve_task(task_name, solver_mode=None, max_iter=100, verbose=True):
    task = TASKS[task_name]
    config = _clone_config(task["config"], solver_mode)
    benchmark_mode = task["benchmark_mode"]
    if benchmark_mode == "direct":
        result = _solve_direct_task(config, max_iter=max_iter, verbose=verbose)
    else:
        result = _solve_wrapper_task(config, max_iter=max_iter, verbose=verbose)
    result["task_name"] = task_name
    result["benchmark_mode"] = benchmark_mode
    result["scene_path"] = getattr(config, "scene_path", task.get("scene_path"))
    return result


def _state_to_mujoco(config, state):
    state = jnp.asarray(state)
    if hasattr(config, "state_to_qpos"):
        return (
            np.asarray(config.state_to_qpos(state)),
            np.asarray(config.state_to_qvel(state)),
        )

    qpos_dim = 7 + config.n_joints
    qvel_start = qpos_dim
    qvel_stop = qvel_start + 6 + config.n_joints
    return (
        np.asarray(state[:qpos_dim]),
        np.asarray(state[qvel_start:qvel_stop]),
    )


def _resolve_base_body_id(config, model):
    body_name = getattr(config, "body_name", None)
    if isinstance(body_name, str):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            return int(body_id)

    if model.nbody > 1:
        return 1
    return 0


def _predicted_base_positions(config, model, qpos_sequence):
    qpos_sequence = np.asarray(qpos_sequence)
    if qpos_sequence.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Floating-base tasks encode the base world position directly in qpos[:3].
    if qpos_sequence.shape[1] >= 7:
        return np.asarray(qpos_sequence[:, :3], dtype=np.float64)

    base_body_id = _resolve_base_body_id(config, model)
    scratch_data = mujoco.MjData(model)
    base_positions = np.zeros((qpos_sequence.shape[0], 3), dtype=np.float64)
    for idx, qpos in enumerate(qpos_sequence):
        scratch_data.qpos = qpos
        mujoco.mj_forward(model, scratch_data)
        base_positions[idx] = np.asarray(scratch_data.xpos[base_body_id], dtype=np.float64)
    return base_positions


def _body_export_layout(task_name, model):
    if task_name == "g1_jump_forward":
        body_names = ISAAC_G1_BODY_NAMES
    else:
        body_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(1, model.nbody)
        )

    body_count = len(body_names)
    body_index_by_name = {name: idx for idx, name in enumerate(body_names) if name}
    mujoco_to_export = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        export_idx = body_index_by_name.get(name)
        if export_idx is not None:
            mujoco_to_export.append((body_id, export_idx))
    return body_names, body_count, mujoco_to_export


def _reorder_named_columns(values, source_names, target_names):
    values = np.asarray(values)
    index_by_name = {name: idx for idx, name in enumerate(source_names)}
    indices = [index_by_name[name] for name in target_names]
    return values[:, indices]


def _play_mujoco_trajectory(result, headless=False, loop=True, ghost_stride=1):
    config = result["config"]
    scene_path = result["scene_path"]
    X = np.asarray(result["X"])
    history = result["history"]

    if headless:
        print(
            "Offline solve shapes:",
            X.shape,
            np.asarray(result["U"]).shape,
            np.asarray(result["reference"]).shape,
            len(history),
        )
        print(
            f"Final objective {result['stats']['final_objective']:.6f} | "
            f"Final dynamics violation {result['stats']['final_dynamics_violation']:.6f}"
        )
        return

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    ghost_geoms = None
    ghost_scratch = mujoco.MjData(model)
    base_marker_ids = None
    plan_states = np.asarray(history[-1][::ghost_stride])
    qpos_sequence = np.asarray(
        [_state_to_mujoco(config, state)[0] for state in plan_states]
    )
    if qpos_sequence.shape[0] == 0:
        qpos_sequence = np.asarray([_state_to_mujoco(config, X[0])[0]])
    display_qpos_sequence = qpos_sequence
    display_alphas = np.ones(display_qpos_sequence.shape[0], dtype=np.float64)
    base_positions = _predicted_base_positions(config, model, display_qpos_sequence)
    base_marker_diameter = float(np.clip(0.03 * model.stat.extent, 0.01, 0.06))

    frame = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            idx = frame % X.shape[0] if loop else min(frame, X.shape[0] - 1)
            qpos, qvel = _state_to_mujoco(config, X[idx])
            data.qpos = qpos
            data.qvel = qvel
            mujoco.mj_forward(model, data)
            ghost_geoms, ghost_scratch = sim_utils.render_ghost_trajectory(
                viewer,
                model,
                display_qpos_sequence,
                display_alphas*0.1,
                ghost_geoms=ghost_geoms,
                scratch_data=ghost_scratch,
                subsample=15,
            )
            base_marker_ids = sim_utils.render_sphere_trajectory(
                viewer,
                base_positions,
                display_alphas,
                diameter=base_marker_diameter,
                color=np.array([1.0, 0.45, 0.0, 1.0]),
                geom_ids=base_marker_ids,
            )
            viewer.sync()
            frame += 1
            time.sleep(config.dt)


def _export_trajectory(result, export_path):
    """Save the solved trajectory and metadata to a .npz file."""

    config = result["config"]
    X = np.asarray(result["X"])

    qpos_list, qvel_list = [], []
    for state in X:
        qpos, qvel = _state_to_mujoco(config, state)
        qpos_list.append(qpos)
        qvel_list.append(qvel)
    qpos_seq = np.asarray(qpos_list)
    qvel_seq = np.asarray(qvel_list)

    joint_pos = np.asarray(qpos_seq[:, 7:], dtype=np.float32)
    joint_vel = np.asarray(qvel_seq[:, 6:], dtype=np.float32)
    joint_names = None
    if result["task_name"] == "g1_jump_forward":
        if joint_pos.shape[1] != len(MPX_G1_JOINT_NAMES):
            raise ValueError(
                f"Expected {len(MPX_G1_JOINT_NAMES)} G1 joints, got {joint_pos.shape[1]}."
            )
        joint_pos = _reorder_named_columns(
            joint_pos, MPX_G1_JOINT_NAMES, ISAAC_G1_JOINT_NAMES
        ).astype(np.float32)
        joint_vel = _reorder_named_columns(
            joint_vel, MPX_G1_JOINT_NAMES, ISAAC_G1_JOINT_NAMES
        ).astype(np.float32)
        joint_names = np.asarray(ISAAC_G1_JOINT_NAMES)

    dt = float(config.dt)
    t = np.arange(X.shape[0], dtype=np.float64) * dt
    fps = int(round(1.0 / dt)) if dt > 0.0 else 0

    model = mujoco.MjModel.from_xml_path(result["scene_path"])
    kin_data = mujoco.MjData(model)
    body_names, body_count, mujoco_to_export = _body_export_layout(result["task_name"], model)
    body_pos_w = np.zeros((qpos_seq.shape[0], body_count, 3), dtype=np.float32)
    body_quat_w = np.zeros((qpos_seq.shape[0], body_count, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0
    body_lin_vel_w = np.zeros((qpos_seq.shape[0], body_count, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((qpos_seq.shape[0], body_count, 3), dtype=np.float32)
    for i in range(qpos_seq.shape[0]):
        kin_data.qpos = qpos_seq[i]
        kin_data.qvel = qvel_seq[i]
        mujoco.mj_forward(model, kin_data)
        cvel = np.asarray(kin_data.cvel, dtype=np.float32)
        for body_id, export_idx in mujoco_to_export:
            body_pos_w[i, export_idx] = np.asarray(kin_data.xpos[body_id], dtype=np.float32)
            body_quat_w[i, export_idx] = np.asarray(kin_data.xquat[body_id], dtype=np.float32)
            body_ang_vel_w[i, export_idx] = cvel[body_id, :3]
            body_lin_vel_w[i, export_idx] = cvel[body_id, 3:]

    export_path = os.path.abspath(export_path)
    os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)

    payload = {
        "dt": np.array(dt),
        "fps": np.array(fps),
        "t": t,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "body_names": np.asarray(body_names),
    }
    if joint_names is not None:
        payload["joint_names"] = joint_names

    np.savez(export_path, **payload)
    print(f"Saved trajectory to {export_path}")


def run_task(
    task_name,
    solver_mode=None,
    headless=False,
    max_iter=100,
    verbose=True,
    loop=True,
    export=None,
):
    result = solve_task(
        task_name,
        solver_mode=solver_mode,
        max_iter=max_iter,
        verbose=verbose,
    )
    stats = result["stats"]
    print(
        f"{task_name} | {result['config'].solver_mode} | "
        f"iterations {stats['n_iterations']} | "
        f"avg iter time {stats['average_iteration_time_ms']:.3f} ms"
    )
    if export is not None:
        _export_trajectory(result, export)
    _play_mujoco_trajectory(result, headless=headless, loop=loop)


def build_parser(default_task=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=tuple(TASKS.keys()),
        default=default_task,
    )
    parser.add_argument(
        "--solver",
        choices=SOLVERS,
        default=None,
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument(
        "--export",
        metavar="PATH",
        default=None,
        help="Save the solved trajectory to the given .npz file.",
    )
    return parser


def main(default_task=None):
    parser = build_parser(default_task=default_task)
    args = parser.parse_args()
    if args.task is None:
        parser.error("--task is required")
    run_task(
        args.task,
        solver_mode=args.solver,
        headless=args.headless,
        max_iter=args.max_iter,
        verbose=not args.quiet,
        loop=not args.no_loop,
        export=args.export,
    )


if __name__ == "__main__":
    main()
