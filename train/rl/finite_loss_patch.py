"""Mask non-finite log-probability tokens out of the policy loss.

Long bf16 packed forwards can produce isolated non-finite response log
probabilities. The underlying clamped loss already gives those tokens zero
gradient; removing them from ``response_mask`` also keeps aggregate loss metrics
finite. ``nonfinite_response_tokens`` reports how many tokens were removed.

Non-invasive: rebinds POLICY_LOSS_REGISTRY["vanilla"] (losses.py:103 resolves
via get_policy_loss_fn at call time). Installed from memory_hf_registry
(external_lib -> FSDP WorkerDict, the process that runs update_actor).
Idempotent.
"""
from __future__ import annotations


def install() -> None:
    try:
        import torch
        from verl.trainer.ppo import core_algos
    except Exception as e:  # pragma: no cover
        print(f"[finite-loss-patch] verl import failed, not patching: {e}", flush=True)
        return
    if core_algos.POLICY_LOSS_REGISTRY.get("vanilla_orig") is not None:
        return
    orig = core_algos.POLICY_LOSS_REGISTRY["vanilla"]

    def vanilla_finite(old_log_prob, log_prob, advantages, response_mask, *args, **kwargs):
        finite = torch.isfinite(old_log_prob) & torch.isfinite(log_prob)
        bad = (~finite & response_mask.bool()).sum()
        if bool(bad):
            response_mask = response_mask * finite
        loss, metrics = orig(old_log_prob, log_prob, advantages, response_mask, *args, **kwargs)
        metrics["actor/nonfinite_response_tokens"] = float(bad)
        return loss, metrics

    core_algos.POLICY_LOSS_REGISTRY["vanilla_orig"] = orig
    core_algos.POLICY_LOSS_REGISTRY["vanilla"] = vanilla_finite
    print("[finite-loss-patch] patched POLICY_LOSS_REGISTRY['vanilla']: non-finite "
          "logprob tokens masked out of the loss (+nonfinite_response_tokens metric)",
          flush=True)
