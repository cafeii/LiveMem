"""DAPO dynamic-sampling group filter for verl fully_async_policy.

GRPO groups with identical rewards have zero advantage. This module filters
such groups before they enter the training queue by patching
``MessageQueueClient.put_sample``. Metrics are read from ``non_tensor_batch``
with ``rm_scores`` as a fallback; samples pass through unchanged when no metric
can be extracted.

Drop counts are bounded by both the DAPO oversampling limit and the asynchronous
staleness window so the rollouter cannot starve the trainer before parameter
synchronization. Non-byte payloads and termination signals always pass through.
The installation is idempotent and active only when group filtering is enabled.
"""
from __future__ import annotations

import numpy as np

DEFAULT_METRIC = "acc"
STD_EPS = 1e-6
PRINT_EVERY = 50

_installed = False
_filter = None  # singleton _GroupFilter, created by install()


def _finite_1d(x) -> np.ndarray | None:
    """np.float64 1-D vector or None (None entries become nan -> rejected)."""
    try:
        vals = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vals.ndim != 1 or vals.size == 0 or not np.isfinite(vals).all():
        return None
    return vals


class _GroupFilter:
    """Filtering core. Pure CPU logic (no ray/verl imports) for unit tests.

    Not thread-safe by design: put_sample runs on the rollouter actor's single
    asyncio loop and all mutation happens before any await.
    """

    def __init__(
        self,
        metric: str,
        required_samples: int,
        trigger_sync_step: int,
        max_num_gen_batches: int,
        max_required_fallback: int,
    ):
        self.metric = metric
        # trainer's per-sync-window consumption, in groups (= queue items)
        self.keep_floor = required_samples * trigger_sync_step
        # Guard A: DAPO max_num_gen_batches semantics; non-positive = no limit
        self.consec_limit = (
            max(max_num_gen_batches - 1, 0) * required_samples if max_num_gen_batches > 0 else None
        )
        self.max_required_fallback = max_required_fallback
        # counters
        self.groups_seen = 0
        self.groups_kept = 0
        self.forced_pass = 0
        self.consecutive_drops = 0
        self.dropped_in_window = 0
        self._last_staleness = None
        self._warned: set[str] = set()

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[group-filter] {msg}", flush=True)

    def group_values(self, full_batch) -> np.ndarray | None:
        """Per-row reward metric: non_tensor[metric] -> non_tensor['score'] -> rm_scores.sum(-1)."""
        ntb = getattr(full_batch, "non_tensor_batch", None)
        if ntb is not None:
            for key in (self.metric, "score"):
                try:
                    if key in ntb:
                        vals = _finite_1d(ntb[key])
                        if vals is not None:
                            return vals
                except Exception:
                    pass
        batch = getattr(full_batch, "batch", None)
        try:
            if batch is not None and "rm_scores" in batch.keys():
                seq = batch["rm_scores"].sum(-1)  # torch tensor; score sits on last response token
                return _finite_1d(seq.detach().float().cpu().numpy())
        except Exception:
            pass
        return None

    def decide(self, full_batch, rollout_status) -> tuple[bool, str]:
        """One group in, (keep, reason) out. reason: informative|no_metric|forced_consec|forced_budget|dropped."""
        self.groups_seen += 1
        status = rollout_status if isinstance(rollout_status, dict) else {}

        # Window boundary: the staleness counter only ever decreases at
        # reset_staleness (param sync) -> reset the per-window drop budget.
        s = status.get("count/staleness_samples")
        if isinstance(s, (int, float)):
            if self._last_staleness is not None and s < self._last_staleness:
                self.dropped_in_window = 0
            self._last_staleness = s

        vals = self.group_values(full_batch)
        if vals is None:
            keep, reason = True, "no_metric"
            self._warn_once(
                "no_metric",
                f"no usable reward metric ('{self.metric}'/'score'/rm_scores) on sample; "
                "passing groups through unfiltered (fail-open)",
            )
        elif float(np.std(vals)) > STD_EPS:
            keep, reason = True, "informative"
        else:
            # zero-advantage group — drop unless a starvation guard fires
            m = status.get("static/max_required_samples")
            if not isinstance(m, (int, float)) or m <= 0:
                m = self.max_required_fallback
            budget = max(int(m) - self.keep_floor, 0)
            if self.consec_limit is not None and self.consecutive_drops >= self.consec_limit:
                keep, reason = True, "forced_consec"
                self.forced_pass += 1
            elif self.dropped_in_window >= budget:
                keep, reason = True, "forced_budget"
                self.forced_pass += 1
            else:
                keep, reason = False, "dropped"

        if keep:
            self.groups_kept += 1
            self.consecutive_drops = 0
        else:
            self.consecutive_drops += 1
            self.dropped_in_window += 1

        if self.groups_seen % PRINT_EVERY == 0:
            print(
                f"[group-filter] seen={self.groups_seen} kept={self.groups_kept} "
                f"dropped={self.groups_seen - self.groups_kept} forced={self.forced_pass} "
                f"kept_ratio={self.groups_kept / self.groups_seen:.3f} "
                f"window_drops={self.dropped_in_window}",
                flush=True,
            )
        return keep, reason

    def metrics(self) -> dict:
        return {
            "filter/groups_seen": self.groups_seen,
            "filter/groups_kept": self.groups_kept,
            "filter/kept_ratio": self.groups_kept / max(self.groups_seen, 1),
            "filter/forced_pass": self.forced_pass,
        }


def install(config) -> None:
    """Patch MessageQueueClient.put_sample. `config` is the ROOT hydra config.

    No-op unless algorithm.filter_groups.enable AND an async_training section
    exists (i.e. the fully_async path). Idempotent.
    """
    global _installed, _filter
    if _installed:
        return
    from omegaconf import OmegaConf

    fg = OmegaConf.select(config, "algorithm.filter_groups")
    if fg is None or not fg.get("enable", False):
        return
    if OmegaConf.select(config, "async_training") is None:
        return  # colocate path: dapo_fit owns filtering there

    try:
        required = int(config.actor_rollout_ref.actor.ppo_mini_batch_size) * int(
            config.async_training.require_batches
        )
        trigger = int(config.async_training.trigger_parameter_sync_step)
        staleness = float(config.async_training.staleness_threshold)
        max_required_fb = int(required * (staleness + 1) * trigger)
    except Exception as e:
        required, trigger, max_required_fb = 64, 1, 128
        print(
            f"[group-filter] cannot derive required_samples from config ({e}); "
            f"falling back to required={required} trigger={trigger} max_required={max_required_fb}",
            flush=True,
        )
    metric = fg.get("metric", None) or DEFAULT_METRIC
    max_gen = int(fg.get("max_num_gen_batches", 0) or 0)

    from ray import cloudpickle

    from verl.experimental.fully_async_policy import message_queue as mq

    if getattr(mq.MessageQueueClient, "_mem_group_filter_installed", False):
        _installed = True
        return

    _filter = _GroupFilter(
        metric=metric,
        required_samples=required,
        trigger_sync_step=trigger,
        max_num_gen_batches=max_gen,
        max_required_fallback=max_required_fb,
    )
    orig_put_sample = mq.MessageQueueClient.put_sample

    async def put_sample(self, sample):
        # Forward termination signals and unsupported payloads unchanged.
        if not isinstance(sample, (bytes, bytearray)):
            return await orig_put_sample(self, sample)
        try:
            rs = cloudpickle.loads(sample)
            full_batch = rs.full_batch
            rollout_status = rs.rollout_status
        except Exception as e:
            _filter._warn_once("unpickle", f"cannot inspect sample ({e}); passing through")
            return await orig_put_sample(self, sample)

        keep, _reason = _filter.decide(full_batch, rollout_status)
        if not keep:
            # Drop the group but return True so the caller
            # (fully_async_rollouter.py:946-947) counts it in
            # total_generated_samples (compute was spent) instead of
            # does not count it as a stale sample.
            return True
        try:
            if isinstance(rollout_status, dict):
                rollout_status.update(_filter.metrics())
                sample = cloudpickle.dumps(rs)
        except Exception:
            pass  # Metrics are optional; preserve the original payload on failure.
        return await orig_put_sample(self, sample)

    mq.MessageQueueClient.put_sample = put_sample
    mq.MessageQueueClient._mem_group_filter_installed = True
    _installed = True
    print(
        f"[group-filter] installed: metric={metric} std_eps={STD_EPS} "
        f"required_samples={required} keep_floor(required*trigger)={_filter.keep_floor} "
        f"window_drop_budget~={max(max_required_fb - _filter.keep_floor, 0)} "
        f"consec_limit={_filter.consec_limit} (MessageQueueClient.put_sample patched)",
        flush=True,
    )


def install_from_data_config(data_config) -> None:
    """Entry for MemoryRLDataset.__init__ — the only hook that receives config.

    create_rl_dataset passes config.data (main_ppo.py:337-343), which is still
    attached to the root DictConfig, so climb to the root to reach
    algorithm.filter_groups / async_training / actor.
    """
    if _installed:
        return
    try:
        root = data_config._get_root()
    except Exception as e:
        print(f"[group-filter] cannot reach root config from data config ({e}); not installing", flush=True)
        return
    install(root)
