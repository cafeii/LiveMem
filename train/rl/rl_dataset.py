"""Custom RLHFDataset that installs the required runtime integrations.

It is functionally identical to verl's RLHFDataset. The driver imports this
module during dataset construction, so the
flash_attn.bert_padding shim gets installed in that process (TaskRunner runs
left_right_2_no_padding but imports neither model.external_lib nor reward.py —
those load in worker / reward-loop processes).
"""
import pathlib
import sys

WS = pathlib.Path(__file__).resolve().parents[2]
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))

from train.rl.flash_attn_shim import install as _install_flash_attn_shim

_install_flash_attn_shim()

# DAPO dynamic sampling (fit patch; active only with algorithm.filter_groups.enable)
from train.rl.dapo_fit import install as _install_dapo_fit

_install_dapo_fit()

# Rollout routing: group-affinity + budget-weighted load balancer. This module
# is imported (via data.custom_cls) in every process that creates the
# GlobalRequestLoadBalancer actor, strictly before creation: main_ppo TaskRunner
# (main_ppo.py:280 dataset < :312 init_workers -> ray_trainer.py:942) and
# FullyAsyncRollouter.__init__ (fully_async_rollouter.py:450 < init_workers ->
# :803). AgentLoopWorkers also load it in __init__ via get_dataset_class
# (agent_loop.py:419), patching the LLMServerClient methods there.
from train.rl.rollout_patch import install as _install_rollout_patch

_install_rollout_patch()

# fully_async DAPO group filter: needs the root hydra config (for
# algorithm.filter_groups etc.), which no module-level import can see — so it
# installs from MemoryRLDataset.__init__ below, which runs inside
# FullyAsyncRollouter.__init__ (fully_async_rollouter.py:450), strictly before
# set_message_queue_client / any put_sample. No-op unless
# algorithm.filter_groups.enable on the fully_async path.
from train.rl.group_filter_patch import install_from_data_config as _install_group_filter

from verl.utils.dataset.rl_dataset import RLHFDataset


class MemoryRLDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, **kwargs):
        _install_group_filter(config)
        super().__init__(data_files, tokenizer, config, processor=processor, **kwargs)
