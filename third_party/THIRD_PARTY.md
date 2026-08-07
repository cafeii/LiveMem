# Third-party components

This directory vendors three external dependencies with their `.git` histories
removed. Their sources and local modifications are documented below.

## flash-linear-attention/

- **Upstream**: https://github.com/fla-org/flash-linear-attention (base version
  `0.5.1`; see `fla/__init__.py`)
- **License**: MIT (see `LICENSE` in this directory)
- **Local modifications**: Adds the following GDN2 operator and layer on top of
  upstream:
  - `fla/ops/gdn2/` — GDN2 kernel implementation
  - `fla/layers/gdn2.py` — `GatedDeltaNet2` layer (base class for
    `models/memory_gdn2.py`)
  - `tests/ops/test_gdn2.py` — GDN2 operator tests
- **Notes**: This fork was maintained as a directory in the original workspace
  and has no independent Git history, so an exact diff against upstream 0.5.1
  was not tracked. No intentional changes were made beyond the GDN2 additions.
  The vendored version must take precedence over any pip-installed `fla`;
  `models/_bootstrap.py` prepends it to `sys.path`.

## delta-Mem/

- **Upstream**: https://github.com/declare-lab/delta-Mem
- **Commit**: `5cd5d9153c7f408764728d953565201e198c39e2` (2026-06-03)
- **Local modifications**: None; vendored as-is.
- **Purpose**:
  - Reproduces the delta-Mem baseline; `eval/delta_server.py` depends on its
    `deltamem` package.
  - Provides the LoCoMo evaluation data at `data/locomo10.json`, read by
    `tools/data_process/preprocess/locomo.py`.

## verl/

- **Upstream**: https://github.com/verl-project/verl
- **Commit**: `7aed6b230776f963fa09509c10d9c3a767d1102c` (2026-06-01, version `0.8.0.dev0`)
- **Local modifications**: None; vendored as-is.
- **Purpose**: Provides the RL (GRPO) training framework. All customizations are
  runtime patches such as `train/rl/*_patch.py`, integrated through verl's
  `data.custom_cls`, `model.external_lib`, `reward.custom_reward_function`, and
  `rollout.agent.agent_loop_config_path` hooks without modifying verl's source.
