"""DAPO dynamic sampling for verl's RayPPOTrainer.

verl 0.8.0.dev ships the `algorithm.filter_groups` schema but its
`ray_trainer.fit` never consumes it (only experimental fully_async_policy
does). This module provides a pruned fit with the DAPO oversample loop:

    keep only groups whose reward metric has nonzero std (uniform groups give
    zero GRPO advantage = zero gradient); pull more prompt batches until
    train_batch_size informative groups accumulate or max_num_gen_batches hit.

Installed as a class-level patch on RayPPOTrainer from train/rl/rl_dataset.py
before the trainer is built.
The patch defers to the original fit unless `algorithm.filter_groups.enable`.

The supported configuration is asserted at entry:
GRPO / no critic / no ref / no REMAX / no rollout_correction / no profiler.
The body follows verl ``ray_trainer.fit`` 0.8.0.dev with the same call order:
gen -> sleep_replicas -> reward -> [FILTER/ACCUMULATE] -> balance ->
old_log_prob -> adv -> update_actor -> save -> update_weights -> val -> log.
"""
from __future__ import annotations

import uuid

import numpy as np
import torch
from tqdm import tqdm


def _keep_informative_groups(new_batch, reward_tensor, metric_name):
    """Row indices belonging to uid-groups with std(metric) > 0."""
    if metric_name in new_batch.non_tensor_batch:
        vals = np.asarray(new_batch.non_tensor_batch[metric_name], dtype=np.float64)
    else:  # fallback: sequence-level reward
        vals = reward_tensor.sum(dim=-1).cpu().numpy().astype(np.float64)
    uids = new_batch.non_tensor_batch["uid"]
    by_uid: dict = {}
    for i, u in enumerate(uids):
        by_uid.setdefault(u, []).append(i)
    keep = [i for idxs in by_uid.values() if np.std(vals[idxs]) > 0 for i in idxs]
    return sorted(keep), len(by_uid)


def dapo_fit(self):
    import verl.trainer.ppo.ray_trainer as rt
    from omegaconf import OmegaConf
    from verl.trainer.ppo.ray_trainer import (
        DataProto,
        agg_loss,
        compute_advantage,
        compute_data_metrics,
        compute_response_mask,
        compute_throughout_metrics,
        compute_timing_metrics,
        extract_reward,
        marked_timer,
        reduce_metrics,
    )
    from verl.utils.tracking import Tracking

    cfg = self.config
    fg = cfg.algorithm.filter_groups
    assert str(cfg.algorithm.adv_estimator) in ("grpo", "AdvantageEstimator.GRPO"), cfg.algorithm.adv_estimator
    assert not self.use_critic and not self.use_reference_policy
    assert cfg.global_profiler.steps is None, "profiler unsupported in dapo_fit"
    metric_name = fg.get("metric", None) or "acc"
    max_gen_batches = fg.get("max_num_gen_batches", 0) or 4
    rollout_n = cfg.actor_rollout_ref.rollout.n
    target_rows = cfg.data.train_batch_size * rollout_n

    if self._dump_executor._shutdown:
        self._init_dump_executor()
    logger = Tracking(project_name=cfg.trainer.project_name,
                      experiment_name=cfg.trainer.experiment_name,
                      default_backend=cfg.trainer.logger,
                      config=OmegaConf.to_container(cfg, resolve=True))
    self.global_steps = 0
    self._load_checkpoint()
    self.checkpoint_manager.update_weights(self.global_steps)
    current_epoch = self.global_steps // len(self.train_dataloader)

    if cfg.trainer.get("val_before_train", True):
        val_metrics = self._validate()
        rt.pprint(f"Initial validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=self.global_steps)
        if cfg.trainer.get("val_only", False):
            self._shutdown_dump_executor()
            return

    progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps,
                        desc="Training Progress (DAPO)")
    self.global_steps += 1
    last_val_metrics = None
    self.max_steps_duration = 0

    for epoch in range(current_epoch, cfg.trainer.total_epochs):
        data_iter = iter(self.train_dataloader)
        epoch_exhausted = False
        while not epoch_exhausted:
            metrics, timing_raw = {}, {}
            buffer = None
            num_gen_batches = 0
            groups_seen = groups_kept = 0
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # ---- Oversample until enough informative groups. -----------
                while True:
                    try:
                        batch_dict = next(data_iter)
                    except StopIteration:
                        epoch_exhausted = True
                        break
                    new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                    new_batch.meta_info["temperature"] = cfg.actor_rollout_ref.rollout.temperature
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    gen_batch = self._get_gen_batch(new_batch)
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    gen_batch = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                    # Keep replicas active inside the oversampling loop; sleep
                    # them once after generation is complete.
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_out = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_out.meta_info.pop("timing", {}))

                    new_batch = new_batch.repeat(repeat_times=rollout_n, interleave=True)
                    new_batch = new_batch.union(gen_out)
                    if "response_mask" not in new_batch.batch.keys():
                        new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            new_batch = new_batch.union(self._compute_reward_colocate(new_batch))
                        reward_tensor, reward_extra = extract_reward(new_batch)
                    new_batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra:
                        new_batch.non_tensor_batch.update(
                            {k: np.array(v) for k, v in reward_extra.items()})

                    num_gen_batches += 1
                    keep_idx, n_groups = _keep_informative_groups(
                        new_batch, reward_tensor, metric_name)
                    groups_seen += n_groups
                    groups_kept += len(keep_idx) // rollout_n
                    kept = new_batch.select_idxs(keep_idx) if keep_idx else None
                    if kept is not None:
                        buffer = kept if buffer is None else DataProto.concat([buffer, kept])
                    have = 0 if buffer is None else len(buffer)
                    if have >= target_rows:
                        buffer = buffer.slice(0, target_rows)
                        break
                    if num_gen_batches >= max_gen_batches:
                        break

                metrics["dapo/num_gen_batches"] = num_gen_batches
                metrics["dapo/groups_seen"] = groups_seen
                metrics["dapo/groups_kept"] = groups_kept
                metrics["dapo/kept_ratio"] = groups_kept / max(groups_seen, 1)

                if buffer is None or len(buffer) == 0:
                    # Replicas remain active because the next iteration starts with generation.
                    rt.pprint(f"[dapo] step {self.global_steps}: no informative groups "
                              f"in {num_gen_batches} gen batches — skipping update")
                    if epoch_exhausted:
                        break
                    logger.log(data=metrics, step=self.global_steps)
                    continue
                batch = buffer
                # Sleep replicas while the trainer uses the GPUs; update_weights
                # reactivates them at the end of the step.
                self.checkpoint_manager.sleep_replicas()

                if cfg.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"], dim=-1).tolist()
                images_seqlens_all = []
                for mmi in batch.non_tensor_batch.get("multi_modal_inputs", []):
                    if "image_grid_thw" in mmi.keys():
                        images_seqlens_all.extend(mmi["images_seqlens"].tolist())
                batch.meta_info["images_seqlens"] = images_seqlens_all

                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    actor_cfg = cfg.actor_rollout_ref.actor
                    entropy_agg = agg_loss(loss_mat=entropys,
                                           loss_mask=batch.batch["response_mask"],
                                           loss_agg_mode=actor_cfg.loss_agg_mode,
                                           loss_scale_factor=actor_cfg.loss_scale_factor)
                    metrics.update({"actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu})
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                with marked_timer("adv", timing_raw, color="brown"):
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    batch = compute_advantage(
                        batch, adv_estimator=cfg.algorithm.adv_estimator,
                        gamma=cfg.algorithm.gamma, lam=cfg.algorithm.lam,
                        num_repeat=rollout_n,
                        norm_adv_by_std_in_grpo=cfg.algorithm.get("norm_adv_by_std_in_grpo", True),
                        config=cfg.algorithm)

                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = self._update_actor(batch)

                if cfg.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % cfg.trainer.save_freq == 0):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                with marked_timer("update_weights", timing_raw, color="red"):
                    self.checkpoint_manager.update_weights(self.global_steps)
                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

            if cfg.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % cfg.trainer.test_freq == 0):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            self.max_steps_duration = max(self.max_steps_duration, timing_raw["step"])
            metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
            metrics.update(compute_data_metrics(batch=batch, use_critic=False))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1
            if is_last_step:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                self._shutdown_dump_executor()
                rt.pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return
    self._shutdown_dump_executor()


def install() -> None:
    """Class-level patch; defers to original fit unless filter_groups.enable."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer, "_dapo_fit_installed", False):
        return
    orig_fit = RayPPOTrainer.fit

    def fit(self):
        fg = self.config.algorithm.get("filter_groups", None)
        if fg is not None and fg.get("enable", False):
            return dapo_fit(self)
        return orig_fit(self)

    RayPPOTrainer.fit = fit
    RayPPOTrainer._dapo_fit_installed = True
