"""Group-affinity and budget-weighted rollout routing for verl GRPO.

Sibling rollouts from one prompt share a stable routing key so they reach the
same vLLM replica and can reuse the memory-prefix cache. The load balancer weights
requests by generation budget and estimated prefill cost rather than request
count alone. Group keys are reference-counted and removed when their requests
finish, while ordinary request keys preserve upstream sticky-routing behavior.

``GroupWeightedLoadBalancer`` follows verl 0.8.0.dev and must be checked when
upgrading verl. ``install()`` is idempotent, and callers that do not support the
weighted signature fall back to upstream unweighted routing.
"""
from __future__ import annotations

import math
import os
from typing import Any, Optional
from uuid import uuid4

from cachetools import LRUCache

GROUP_KEY_PREFIX = "memgrp-"
DEFAULT_ROUTING_CACHE_SIZE = 10000  # keep in sync with llm_server.py:40

_installed = False


def group_routing_key(extra_info: Any) -> Optional[str]:
    """Routing key shared by the rollout.n siblings of one dataset row.

    Uses extra_info.index (globally unique per row in our parquets) rather than
    the top-level kwargs["index"], because verl defaults the latter to 0 for
    every row when extra_info.index is absent (rl_dataset.py:384) — which would
    collapse all traffic onto one server. No index -> None -> caller keeps the
    upstream per-request uuid routing.
    """
    if not isinstance(extra_info, dict):
        return None
    index = extra_info.get("index")
    if index is None:
        return None
    return f"{GROUP_KEY_PREFIX}{index}"


def request_weight(
    prompt_len: int, sampling_params: dict, default_max_tokens: int, prefill_coef: float
) -> int:
    """In-flight cost estimate: decode budget + amortized prefill share."""
    max_tokens = (
        sampling_params.get("max_tokens")
        or sampling_params.get("max_new_tokens")
        or default_max_tokens
    )
    return max(int(max_tokens) + math.ceil(prefill_coef * prompt_len), 1)


class GroupRoutedClient:
    """Per-request proxy that pins routing to a fixed group key.

    Installed by MemorySingleTurnAgentLoop on its own (per-request) instance, so
    every generate() — including FullyAsyncLLMServerClient's partial-rollout
    resume rounds, which reuse the outer request_id — routes by the group key.
    Everything else delegates to the wrapped client.
    """

    __slots__ = ("_inner", "_routing_key")

    def __init__(self, inner, routing_key: str):
        self._inner = inner
        self._routing_key = routing_key

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def generate(self, request_id=None, **kwargs):
        return await self._inner.generate(self._routing_key, **kwargs)


class GroupWeightedLoadBalancer:
    """GlobalRequestLoadBalancer with budget weights + group-key entry lifecycle.

    Plain class without ``@ray.remote``; install() wraps it
    with ray.remote and rebinds llm_server.GlobalRequestLoadBalancer. Body
    transcribed from third_party/verl/verl/workers/rollout/llm_server.py:44-143
    (0.8.0.dev).
    """

    def __init__(self, servers: dict, max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE):
        if not servers:
            raise ValueError("servers must be non-empty")
        self._servers: dict = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        # Track in-flight requests per group and remove sticky routing at zero.
        # The LRU bound limits growth if a release is missing.
        self._group_inflight: LRUCache = LRUCache(maxsize=max_cache_size)

    def acquire_server(self, request_id: str, weight: int = 1):
        weight = max(int(weight), 1)  # weight=1 matches upstream request counting.
        if request_id.startswith(GROUP_KEY_PREFIX):  # Reference-count group keys.
            self._group_inflight[request_id] = self._group_inflight.get(request_id, 0) + 1

        # Try sticky session first
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            if server_id in self._inflight_requests:
                self._inflight_requests[server_id] += weight
                return server_id, self._servers[server_id]
            # Server was removed, clear stale cache entry and re-select
            del self._request_id_to_server[request_id]

        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        server_id = min(self._inflight_requests, key=self._inflight_requests.get)
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += weight
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str, weight: int = 1, request_id: str = None) -> None:
        weight = max(int(weight), 1)
        # Update group-key lifecycle even if the server was removed.
        if request_id is not None and request_id.startswith(GROUP_KEY_PREFIX):
            remaining = self._group_inflight.get(request_id, 0) - 1
            if remaining > 0:
                self._group_inflight[request_id] = remaining
            else:
                self._group_inflight.pop(request_id, None)
                self._request_id_to_server.pop(request_id, None)
        if server_id not in self._inflight_requests:
            return
        # Clamp the weighted decrement at zero, matching upstream behavior.
        self._inflight_requests[server_id] = max(self._inflight_requests[server_id] - weight, 0)

    def add_servers(self, servers: dict) -> None:
        for sid, handle in servers.items():
            self._inflight_requests[sid] = 0
            self._servers[sid] = handle

    def remove_servers(self, server_ids: list) -> None:
        for sid in server_ids:
            self._inflight_requests.pop(sid, None)
            self._servers.pop(sid, None)

    def get_inflight_count(self, server_id: str) -> int:
        return self._inflight_requests.get(server_id, 0)

    def get_all_servers(self) -> list:
        return list(self._inflight_requests.keys())

    def get_status(self) -> dict:
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
            "group_keys_inflight": len(self._group_inflight),
        }


def install() -> None:
    global _installed
    if _installed:
        return
    try:
        import ray
        from verl.utils.rollout_trace import rollout_trace_op
        from verl.workers.rollout import llm_server
    except Exception as e:  # pragma: no cover
        print(f"[rollout-patch] verl/ray import failed, not patching: {e}", flush=True)
        return
    if getattr(llm_server, "_mem_rollout_patched", False):
        _installed = True
        return

    # Rebind the load-balancer class. _init_global_load_balancer and local imports
    # resolve the module global at call time, so every LB actor created
    # after this point runs the weighted class.
    llm_server.GlobalRequestLoadBalancer = ray.remote(GroupWeightedLoadBalancer)

    client_cls = llm_server.LLMServerClient
    # Fall back to upstream unweighted routing if the actor predates this patch.
    client_cls._mem_lb_weighted = True

    async def _acquire_server(self, request_id: str, weight: int = 1):
        if client_cls._mem_lb_weighted:
            try:
                return await self._load_balancer.acquire_server.remote(
                    request_id=request_id, weight=weight
                )
            except TypeError:
                client_cls._mem_lb_weighted = False
                print("[rollout-patch] LB actor is unpatched, falling back to "
                      "unweighted routing", flush=True)
        return await self._load_balancer.acquire_server.remote(request_id=request_id)

    def _release_server(self, server_id: str, weight: int = 1, request_id: str = None) -> None:
        # Fire-and-forget, matching upstream behavior.
        if client_cls._mem_lb_weighted:
            try:
                self._load_balancer.release_server.remote(
                    server_id=server_id, weight=weight, request_id=request_id
                )
                return
            except TypeError:
                client_cls._mem_lb_weighted = False
        self._load_balancer.release_server.remote(server_id=server_id)

    # Based on llm_server.py:179-220 for verl 0.8.0.dev. The only change is
    # pairing a computed weight through acquire/release. Patching the base class
    # ensures FullyAsyncLLMServerClient.generate's
    # super().generate() per resume round lands here (weight then tracks the
    # shrinking remaining max_tokens automatically).
    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        **kwargs,
    ):
        rollout_cfg = self.config.actor_rollout_ref.rollout
        coef_env = os.getenv("MEM_LB_PREFILL_COEF")
        prefill_coef = float(coef_env) if coef_env else 1.0 / max(int(rollout_cfg.n), 1)
        weight = request_weight(
            len(prompt_ids), sampling_params, int(rollout_cfg.response_length), prefill_coef
        )
        server_id, server = await self._acquire_server(request_id, weight=weight)
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            output = await server.generate.remote(
                request_id=uuid4().hex,  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **kwargs,
            )
            return output
        finally:
            self._release_server(server_id, weight=weight, request_id=request_id)

    client_cls._acquire_server = _acquire_server
    client_cls._release_server = _release_server
    client_cls.generate = generate
    llm_server._mem_rollout_patched = True
    _installed = True
    print("[rollout-patch] installed: group-affinity routing + budget-weighted "
          "load balancer (GlobalRequestLoadBalancer rebound, LLMServerClient "
          "acquire/release/generate patched)", flush=True)
