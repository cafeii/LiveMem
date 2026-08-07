"""Per-request generation budgets and group-affinity routing.

The rule table assigns each sample a max_new value by bucket; verl's
response_length is run-global, but the vllm server honors a per-request
`sampling_params["max_tokens"]` (vllm_async_server.py, capped by
response_length). This agent loop injects the bucket budget packed by
to_verl_parquet.py into extra_info.max_new.

For affinity routing, the rollout.n siblings of one prompt
all carry the same extra_info.index, so we route them by a shared group key —
one vllm replica pays the expensive memory-encoding prefill once, the rest hit
its prefix cache. Swapping self.server_manager is per-request-safe because
_run_agent_loop instantiates a fresh agent loop per sample (agent_loop.py:589).

Registered via rollout.agent.agent_loop_config_path (train/rl/agent_loop.yaml)
under TWO names: `memory_single_turn` (rollout.agent.default_agent_loop; used by
main_ppo where agent_name is absent from non_tensor_batch) and
`single_turn_agent` (fully_async_policy force-writes
non_tensor_batch["agent_name"]="single_turn_agent" in detach_utils.py:67, which
bypasses default_agent_loop entirely — without the alias this loop never runs
in fully async deployments).
"""
from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from train.rl.rollout_patch import GroupRoutedClient, group_routing_key
from train.rl.rollout_patch import install as _install_rollout_patch

# Client-side patch (weighted acquire/release) must be live in this
# AgentLoopWorker process before the first generate() call; importing this
# module is a precondition of any generate, so this is structurally early
# enough (rl_dataset.py installs it even earlier, at worker __init__).
_install_rollout_patch()


class MemorySingleTurnAgentLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info")
        extra_info = extra_info if isinstance(extra_info, dict) else {}
        max_new = extra_info.get("max_new")
        if max_new:
            sampling_params = {**sampling_params, "max_tokens": int(max_new)}
        key = group_routing_key(extra_info)
        if key is not None:
            self.server_manager = GroupRoutedClient(self.server_manager, key)
        return await super().run(sampling_params, **kwargs)
