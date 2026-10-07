"""Exercise one LIBERO-Safety task without downloading demonstration datasets.

The evaluation policy deliberately emits zero actions. The collection mode records
those actions through robosuite's DataCollectionWrapper and converts the raw
recording to LIBERO's HDF5 dataset format. It is a pipeline smoke test, not a
successful task demonstration.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "deployment" / ".libero"
RUNS_DIR = ROOT / "deployment" / "runs"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(CONFIG_DIR))
os.environ.setdefault("MUJOCO_GL", "egl")


def prepare_config():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "datasets").mkdir(parents=True, exist_ok=True)
    benchmark_root = ROOT / "libero" / "libero"
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "assets": str(benchmark_root / "assets"),
        "datasets": str(RUNS_DIR / "datasets"),
    }
    with (CONFIG_DIR / "config.yaml").open("w") as stream:
        yaml.safe_dump(paths, stream)


def task_paths(suite_name, level, task_id):
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task_by_level_id(level, task_id)
    if task is None:
        raise ValueError(f"Unknown task: {suite_name} L{level} #{task_id}")
    bddl = Path(suite.get_task_bddl_file_path_by_level_id(level, task_id))
    if not bddl.is_file():
        raise FileNotFoundError(bddl)
    return suite, task, bddl


def evaluate(suite, task, bddl, steps, policy_module, init_index):
    import numpy as np

    from libero.libero.envs import OffScreenRenderEnv

    policy = importlib.import_module(policy_module) if policy_module else None
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=64, camera_widths=64
    )
    try:
        obs = env.reset()
        init_file = (ROOT / "libero" / "libero" / "init_files" / task.problem_folder
                     / f"L{task.level}" / task.init_states_file)
        if init_file.is_file():
            states = suite.get_task_init_states_by_level_id(task.level, task.level_id)
            if not 0 <= init_index < len(states):
                raise ValueError(f"Initial state index {init_index} outside 0..{len(states) - 1}")
            obs = env.set_init_state(states[init_index])
            init_source = "fixed_init_state"
        else:
            if init_index != 0:
                raise ValueError("--init-index requires a published initial-state file")
            init_source = "random_reset"
        image = obs.get("agentview_image")
        if image is None or image.shape[:2] != (64, 64):
            raise RuntimeError(f"Missing agentview image: {None if image is None else image.shape}")
        success = bool(env.check_success())
        for _ in range(steps):
            action = np.asarray(
                policy.act(obs, task.language) if policy else np.zeros(7),
                dtype=np.float32,
            )
            if action.shape != (7,):
                raise ValueError(f"Policy action must have shape (7,), got {action.shape}")
            obs, _, _, _ = env.step(action)
            success = success or bool(env.check_success())
        result = {"suite": suite.name, "level": task.level, "task": task.name,
                  "steps": steps, "success": success, "image_shape": list(image.shape),
                  "policy": policy_module or "zero_action", "init_index": init_index,
                  "init_source": init_source}
        run_dir = RUNS_DIR / datetime.now().strftime("eval_%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"result_file={run_dir / 'result.json'}")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


def collect(task, bddl, steps):
    import h5py
    import numpy as np
    import robosuite as suite
    from robosuite.wrappers import DataCollectionWrapper

    from libero.libero.envs import TASK_MAPPING
    import libero.libero.envs.bddl_utils as bddl_utils

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_dir = run_dir / "raw"
    out_dir = run_dir / "demo"
    out_dir.mkdir(parents=True)
    problem_info = bddl_utils.get_problem_info(str(bddl))
    controller = suite.load_controller_config(default_controller="OSC_POSE")
    env_info = json.dumps({"robots": ["Panda"], "controller_configs": controller})
    env = TASK_MAPPING[problem_info["problem_name"]](
        bddl_file_name=str(bddl),
        robots=["Panda"],
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
    )
    wrapped = DataCollectionWrapper(env, str(raw_dir))
    try:
        wrapped.reset()
        for _ in range(steps):
            wrapped.step(np.zeros(env.action_dim, dtype=np.float32))
    finally:
        wrapped.close()

    # Import the repository's own raw-NPZ -> HDF5 collector.
    sys.path.insert(0, str(ROOT / "scripts"))
    from collect_demonstration import gather_demonstrations_as_hdf5

    gather_demonstrations_as_hdf5(
        str(raw_dir), str(out_dir), env_info,
        SimpleNamespace(bddl_file=str(bddl)), problem_info,
    )
    raw_hdf5 = out_dir / "demo.hdf5"
    with h5py.File(raw_hdf5) as recording:
        episodes = list(recording["data"])
        if len(episodes) != 1:
            raise RuntimeError(f"Expected one episode, got {episodes}")
        actions = recording[f"data/{episodes[0]}/actions"]
        if len(actions) != steps:
            raise RuntimeError(f"Expected {steps} actions, got {len(actions)}")

    # Replay the raw recording to create the observation-rich LIBERO dataset.
    config_dir = run_dir / "config"
    config_dir.mkdir()
    with (CONFIG_DIR / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config["datasets"] = str(run_dir / "datasets")
    with (config_dir / "config.yaml").open("w") as stream:
        yaml.safe_dump(config, stream)
    replay_env = os.environ.copy()
    replay_env["LIBERO_CONFIG_PATH"] = str(config_dir)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "create_dataset.py"),
         "--demo-file", str(raw_hdf5), "--use-camera-obs"],
        cwd=ROOT, check=True, timeout=300, env=replay_env,
    )
    dataset = run_dir / "datasets" / task.problem_folder / f"L{task.level}" / f"{task.name}_demo.hdf5"
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset conversion did not create {dataset}")
    with h5py.File(dataset) as converted:
        count = converted["data"].attrs["num_demos"]
        if count != 1:
            raise RuntimeError(f"Expected one converted episode, got {count}")
    print(json.dumps({"raw_hdf5": str(raw_hdf5), "dataset": str(dataset),
                      "episodes": 1, "steps": steps}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["eval", "collect"])
    parser.add_argument("--suite", default="affordance")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--policy-module", help="Module exposing act(observation, instruction) -> 7 actions")
    parser.add_argument("--init-index", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.mode == "collect" and args.steps <= 5:
        parser.error("collection needs at least 6 steps because create_dataset.py skips the first 5")
    prepare_config()
    suite, task, bddl = task_paths(args.suite, args.level, args.task_id)
    if args.mode == "eval":
        evaluate(suite, task, bddl, args.steps, args.policy_module, args.init_index)
    else:
        collect(task, bddl, args.steps)


if __name__ == "__main__":
    main()
