# Local demo deployment

This checkout uses Python 3.10 in `.conda-env`. `deployment/install_demo.sh` reproduces the
minimal simulator, benchmark, collection and replay dependencies. The demo launcher writes
an isolated LIBERO config to `deployment/.libero/config.yaml`; it does not change `~/.libero`.

```bash
cd /home/drone/Desktop/VACodebase/LIBERO-Safety
bash deployment/install_demo.sh
```

Required simulator resources are installed under `libero/libero/assets/`. They come from
[`LIBERO-Safety/libero_safety_assets`](https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets)
(`assets.zip`, about 10.7 GB). The full demonstration dataset and model weights are not
needed for these checks and are not installed here.

Run a one-task evaluation smoke test:

```bash
cd /home/drone/Desktop/VACodebase/LIBERO-Safety
MUJOCO_GL=egl .conda-env/bin/python deployment/smoke_pipeline.py eval \
  --suite affordance --level 0 --task-id 0 --steps 8
```

The default policy emits zero actions and the resulting `success` field is a pipeline
check, not a meaningful task success rate. To test a policy, put a Python module on
`PYTHONPATH` with `act(observation, instruction) -> array_like[7]` and pass
`--policy-module your_module`. The result JSON is saved below `deployment/runs/`.
The `reasoning_safety` tasks have BDDL files but no published initial-state files in
this checkout or the assets archive. The demo runner uses a random reset for those
tasks and marks `init_source` accordingly.

Run the small noninteractive collection and conversion check:

```bash
MUJOCO_GL=egl .conda-env/bin/python deployment/smoke_pipeline.py collect \
  --suite affordance --level 0 --task-id 0 --steps 8
```

It records raw MuJoCo states and actions through robosuite's `DataCollectionWrapper`,
aggregates them with `scripts/collect_demonstration.py`, and replays them with
`scripts/create_dataset.py` into an observation-rich HDF5 file under a timestamped
`deployment/runs/<run-id>/datasets/` directory. The zero-action sample
is intentionally not a successful demonstration.

For human collection, use a GUI session and a keyboard or SpaceMouse. For example:

```bash
LIBERO_CONFIG_PATH="$PWD/deployment/.libero" \
  .conda-env/bin/python scripts/collect_demonstration.py \
  --bddl-file "$PWD/libero/libero/bddl_files/affordance/L0/grab_the_knife_and_cut_the_lemon_on_the_cutting_board.bddl" \
  --device keyboard --num-demonstration 1 --directory deployment/runs/human
```

The official README says the separate keypose-driven automatic **data generation**
pipeline has not yet been open-sourced. The available manual collector and replay
converter are the data-collection path verified here.
