"""Official-style AMAGO configuration, dataset, and Experiment composition."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

import amago
import gin
import numpy as np
import torch
from amago import utils as amago_utils
from amago.envs import AMAGOEnv
from amago.loading import RLData_pad_collate
from torch import nn
from torch.utils.data import DataLoader

from reasoned_icrl.experiments.artifacts import (
    AMAGO_CONFIG_FILE,
    COLLECTION_COUNTERS,
    TRAINING_METRICS_FILE,
)
from reasoned_icrl.experiments.contracts import (
    ContractError,
    architecture_uses_history_packet,
)
from reasoned_icrl.experiments.xland_one_rule import CurriculumSchedule
from reasoned_icrl.model.utils import freeze_unoptimized_parameters
from reasoned_icrl.runtime.amago import (
    AMAGOComponents,
    configure_amago,
)
from reasoned_icrl.runtime.checkpointing import (
    AMAGORuntimeState,
    _sequence_wrappers,
    policy_checkpoint,
    read_policy_checkpoint,
    register_runtime_state,
    restore_reproduction_material,
    snapshot_reproduction_material,
    validate_checkpoint_architecture,
)
from reasoned_icrl.runtime.environments import benchmark_relabeler
from reasoned_icrl.runtime.replay import (
    OrderedDiskTrajDataset,
    create_replay_dataset,
)


def _sum_counters(reported: Any) -> dict[str, int]:
    """Total one counter snapshot across the parallel actors.

    Each actor reports cumulative counts, so the totals are the sum and no
    running accumulation is needed. An actor that predates the counters
    contributes nothing rather than failing the epoch."""
    totals = dict.fromkeys(COLLECTION_COUNTERS, 0)
    for entry in reported if isinstance(reported, (list, tuple)) else []:
        if not isinstance(entry, Mapping):
            continue
        for key in COLLECTION_COUNTERS:
            totals[key] += int(entry.get(key, 0))
        # Environment-specific exposure counters (the one-rule curriculum's
        # per-pool tasks and calls) travel beside the shared ones.
        for key, value in entry.items():
            if key not in COLLECTION_COUNTERS and type(value) is int:
                totals[key] = totals.get(key, 0) + value
    return totals


class ReasonedExperiment(amago.Experiment):
    """AMAGO v3.4 Experiment with loadable shared-tensor training states."""

    encoder_architecture_id: str
    #: The Weights & Biases run name (``<protocol>/<condition>/seed-<n>``),
    #: tags and extra config, set by :func:`create_experiment`. AMAGO would
    #: otherwise name every run after its directory, ``seed-<n>``. Tags carry
    #: the condition alone so a tag filter selects exactly one matrix cell;
    #: protocol, seed and architecture stay queryable in ``wandb_config``.
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_config: Mapping[str, object] = {}

    def init_logger(self) -> None:
        """AMAGO's logger init with the run named by protocol, condition and seed."""
        if not self.log_to_wandb:
            super().init_logger()
            return
        gin_config = gin.operative_config_str()
        with open(os.path.join(self.ckpt_dir, "config.txt"), "w") as handle:
            handle.write(gin_config)
        config = dict(amago_utils.gin_as_wandb_config())
        config.update(self.wandb_config)
        log_dir = os.path.join(self.ckpt_base_dir, self.run_name, "wandb_logs")
        os.makedirs(log_dir, exist_ok=True)
        self.accelerator.init_trackers(
            project_name=self.wandb_project,
            config=config,
            init_kwargs={
                "wandb": {
                    "entity": self.wandb_entity,
                    "dir": log_dir,
                    "name": self.wandb_run_name or self.run_name,
                    "group": self.wandb_group_name,
                    "tags": list(self.wandb_tags),
                }
            },
        )

    @property
    def uses_history_packet(self) -> bool:
        """Whether this run's encoder consumes history packets."""
        return architecture_uses_history_packet(str(self.encoder_architecture_id))

    def init_optimizer(self, policy: Any) -> torch.optim.Optimizer:
        optimizer = cast(torch.optim.Optimizer, super().init_optimizer(policy))
        self.frozen_unoptimized_parameters = freeze_unoptimized_parameters(
            policy,
            (
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ),
        )
        self.learner_contract = "amago-optimizer-ownership.v1"
        return optimizer

    def caster(self) -> AbstractContextManager[Any]:
        """Use the declared history precision for collection and cache refresh."""
        if self.uses_history_packet and self.mixed_precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return cast(AbstractContextManager[Any], super().caster())

    def init_dloaders(self) -> DataLoader:
        """Create the replay loader with pinned memory only on CUDA."""
        train_dloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            collate_fn=RLData_pad_collate,
            pin_memory=self.DEVICE.type == "cuda",
        )
        self.train_dloader = self.accelerator.prepare(train_dloader)
        return self.train_dloader

    def log(
        self, metrics_dict: dict[str, torch.Tensor | int | float], key: str
    ) -> None:
        """Mirror AMAGO metrics to an append-only local JSONL audit trail."""
        super().log(metrics_dict, key)
        if not self.accelerator.is_main_process:
            return
        row: dict[str, str | int | float | None] = {"panel": key}
        for name, value in self.x_axis_metrics().items():
            scalar = value.item() if isinstance(value, np.generic) else value
            row[name] = (
                None
                if isinstance(scalar, float) and not math.isfinite(scalar)
                else scalar
            )
        if getattr(self, "reasoned_environment_name", None) == "match_pattern":
            row.update(
                _sum_counters(
                    amago_utils.call_async_env(self.train_envs, "collection_counters")
                )
            )
        for name, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    continue
                scalar = value.detach().cpu().float().item()
            elif isinstance(value, np.generic):
                scalar = cast(int | float, value.item())
            elif isinstance(value, int | float):
                scalar = value
            else:
                continue
            row[name] = (
                None
                if isinstance(scalar, float) and not math.isfinite(scalar)
                else scalar
            )
        metrics_path = Path(self.local_metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    def collect_new_training_data(self) -> None:
        """Collect replay and retain AMAGO's rollout metrics locally."""
        collection_started = time.perf_counter()
        if (
            self.force_reset_train_envs_every is not None
            and self.epoch % self.force_reset_train_envs_every == 0
        ):
            self.train_envs.reset()
            self.hidden_state = None
        if self.uses_history_packet and self.DEVICE.type == "cuda":
            torch.cuda.synchronize(self.DEVICE)
        refresh_started = time.perf_counter()
        if self.uses_history_packet:
            self._refresh_history_cache()
        if self.uses_history_packet and self.DEVICE.type == "cuda":
            torch.cuda.synchronize(self.DEVICE)
        refresh_seconds = time.perf_counter() - refresh_started
        self.hidden_state, (returns, specials) = self.interact(
            self.train_envs,
            self.train_timesteps_per_epoch,
            hidden_state=self.hidden_state,
            sample=self.sample_actions_train,
        )
        amago_utils.call_async_env(self.train_envs, "save_finished_trajs")
        self.collection_counters = _sum_counters(
            amago_utils.call_async_env(self.train_envs, "collection_counters")
        )
        if getattr(self, "reasoned_environment_name", None) == "match_pattern":
            bitmaps = amago_utils.call_async_env(self.train_envs, "exposure_bitmap")
            packed = np.stack(
                [np.frombuffer(value, dtype=np.uint8) for value in bitmaps]
            )
            self.collection_counters["unique_completed_examples"] = int(
                np.unpackbits(np.bitwise_or.reduce(packed, axis=0)).sum()
            )
            self.collection_counters["completed_decisions"] = self.collection_counters[
                "tasks_completed"
            ]
            self.collection_counters["forced_advances"] = (
                self.collection_counters["charged_calls"]
                - self.collection_counters["completed_decisions"]
            )
        self.log(self.policy_metrics(returns, specials), key="train-rollout")
        if self.uses_history_packet:
            if not hasattr(self, "history_epoch_timings"):
                self.history_epoch_timings: list[dict[str, float]] = []
            self.history_epoch_timings.append(
                {
                    "collection_seconds": time.perf_counter() - collection_started,
                    "cache_refresh_seconds": refresh_seconds,
                    "update_seconds": 0.0,
                    "updates": 0.0,
                }
            )

    def _refresh_history_cache(self) -> None:
        """Rebuild processed prefixes under current weights, without action sampling.

        Encode the actor batch together. Temporary suffix padding only fills
        unused cache entries, which are erased before returning to interaction.
        """
        policy = self.policy
        policy.eval()
        carrier = policy.traj_encoder
        hidden = carrier.init_hidden_state(self.parallel_actors, self.DEVICE)
        sequences = [
            sequence.active_trajs[0].as_input_sequence()
            for sequence in _sequence_wrappers(self.train_envs, self.env_mode)
        ]
        lengths = [time.shape[1] - 1 for _, _, time in sequences]
        count = max(lengths)
        if count:
            observation = {
                key: np.zeros(
                    (len(sequences), count, *value.shape[2:]), dtype=value.dtype
                )
                for key, value in sequences[0][0].items()
            }
            feedback = np.zeros(
                (len(sequences), count, sequences[0][1].shape[-1]), dtype=np.float32
            )
            positions = np.zeros((len(sequences), count, 1), dtype=np.int64)
            for row, ((obs, rl2, time), length) in enumerate(
                zip(sequences, lengths, strict=True)
            ):
                if int(time[0, 0, 0]) != 0:
                    raise ContractError(
                        "Live history rollout lost its outer-task prefix."
                    )
                for key, value in obs.items():
                    observation[key][row, :length] = value[0, :length]
                feedback[row, :length] = rl2[0, :length]
                positions[row, :length] = time[0, :length]
            with torch.no_grad(), self.caster():
                packet = policy.tstep_encoder(
                    {
                        key: torch.as_tensor(value, device=self.DEVICE)
                        for key, value in observation.items()
                    },
                    torch.as_tensor(feedback, device=self.DEVICE),
                )
                time_tensor = torch.as_tensor(positions, device=self.DEVICE)
                rebuild = getattr(carrier, "rebuild_hidden_state", None)
                if rebuild is not None:
                    # Relational and symbol state cannot be repaired after the
                    # fact, so a carrier that owns such state rebuilds true
                    # prefixes itself rather than processing padded decisions
                    # and having its ordinary lengths reset afterwards.
                    self.hidden_state = rebuild(packet, time_tensor, lengths)
                    return
                token_dim = carrier.token_dim
                for index in range(count):
                    _, hidden = carrier.backbone(
                        packet[:, index : index + 1, :token_dim],
                        time_tensor[:, index : index + 1],
                        hidden,
                    )
            for row, length in enumerate(lengths):
                hidden.key_cache.data[:, row, length:] = torch.nan
                hidden.val_cache.data[:, row, length:] = torch.nan
                hidden.seq_lens[row] = length
        self.hidden_state = hidden

    def train_step(self, batch: Any, log_step: bool) -> dict[str, Any]:
        if self.uses_history_packet and not getattr(
            self, "history_epoch_timings", None
        ):
            # Native train_step is also used for a restored replay-only update.
            # Logging must not require a preceding environment collection call.
            self.history_epoch_timings = [
                {"collection_seconds": 0.0, "update_seconds": 0.0, "updates": 0.0}
            ]
        started = time.perf_counter()
        metrics = cast(dict[str, Any], super().train_step(batch, log_step))
        if self.uses_history_packet:
            if self.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            self.history_epoch_timings[-1]["update_seconds"] += (
                time.perf_counter() - started
            )
            self.history_epoch_timings[-1]["updates"] += 1
        if self.uses_history_packet and not torch.isfinite(metrics["Loss"]).all():
            raise ContractError("Nonfinite history training loss; run stopped.")
        return metrics

    def save_checkpoint(self) -> None:
        """Save Accelerate state without SafeTensors' shared-storage rejection."""
        if self.uses_history_packet and any(
            not torch.isfinite(value).all()
            for value in self.policy.state_dict().values()
        ):
            raise ContractError("Nonfinite history policy checkpoint; run stopped.")
        checkpoint_started = time.perf_counter()
        checkpoint_name = f"{self.run_name}_epoch_{self.epoch}"
        self.accelerator.save_state(
            os.path.join(self.ckpt_dir, "training_states", checkpoint_name),
            safe_serialization=False,
        )
        if self.accelerator.is_main_process:
            torch.save(
                policy_checkpoint(
                    self.policy.state_dict(),
                    condition=self.policy_condition,
                    architecture_id=str(self.encoder_architecture_id),
                ),
                os.path.join(
                    self.ckpt_dir,
                    "policy_weights",
                    f"policy_epoch_{self.epoch}.pt",
                ),
            )

        if getattr(self, "reasoned_environment_name", None) == "match_pattern":
            import hashlib

            weights = (
                Path(self.ckpt_dir) / "policy_weights" / f"policy_epoch_{self.epoch}.pt"
            )
            metadata = {
                "label": int(self.epoch),
                **self.collection_counters,
                "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "packet_sha256": self.policy.tstep_encoder.spec.sha256,
            }
            temporary = weights.with_suffix(".metadata.tmp")
            temporary.write_text(json.dumps(metadata, sort_keys=True) + "\n")
            temporary.replace(weights.with_suffix(".metadata.json"))

        if getattr(self, "reasoned_training_settings", {}).get("retain_training_state"):
            snapshot_reproduction_material(
                Path(self.ckpt_dir) / "training_states" / checkpoint_name,
                self.reasoned_dataset,
                Path(self.local_metrics_path),
            )

        self.checkpoint_seconds = getattr(self, "checkpoint_seconds", 0.0) + (
            time.perf_counter() - checkpoint_started
        )

    def evaluate_val(self) -> None:
        started = time.perf_counter()
        super().evaluate_val()
        if self.DEVICE.type == "cuda":
            torch.cuda.synchronize(self.DEVICE)
        self.validation_seconds = getattr(self, "validation_seconds", 0.0) + (
            time.perf_counter() - started
        )
        # R4: validation interactions are charged to their own clock, never to
        # the training collection budget; the actors report cumulative totals.
        self.validation_counters = _sum_counters(
            amago_utils.call_async_env(self.val_envs, "collection_counters")
        )

    def load_checkpoint(self, epoch: int, resume_training_state: bool = True) -> None:
        """Preflight the project runtime contract before Accelerate mutates state."""
        adapter = getattr(self, "reasoned_runtime_state", None)
        if resume_training_state:
            if not isinstance(adapter, AMAGORuntimeState):
                raise ContractError(
                    "Training resume requires the runtime-state adapter."
                )
            checkpoint_name = f"{self.run_name}_epoch_{epoch}"
            checkpoint_root = Path(self.ckpt_dir) / "training_states" / checkpoint_name
            runtime_states: list[Mapping[str, object]] = []
            for checkpoint in sorted(checkpoint_root.glob("custom_checkpoint_*.pkl")):
                raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if isinstance(raw, Mapping) and str(raw.get("schema", "")).startswith(
                    "amago-runtime-state."
                ):
                    runtime_states.append(cast(Mapping[str, object], raw))
            if len(runtime_states) != 1:
                raise ContractError(
                    "AMAGO checkpoint requires exactly one runtime state."
                )
            adapter.validate_state_dict(runtime_states[0])
            if getattr(self, "reasoned_training_settings", {}).get(
                "retain_training_state"
            ):
                restore_reproduction_material(
                    checkpoint_root,
                    self.reasoned_dataset,
                    Path(self.local_metrics_path),
                )
        elif not resume_training_state:
            checkpoint = (
                Path(self.ckpt_dir) / "policy_weights" / f"policy_epoch_{epoch}.pt"
            )
            raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
            raw = read_policy_checkpoint(
                raw,
                condition=self.policy_condition,
                architecture_id=str(self.encoder_architecture_id),
            )
            validate_checkpoint_architecture(
                raw,
                str(self.encoder_architecture_id),
                expected_state=self.policy.state_dict(),
            )
            self.policy.load_state_dict(raw)
            return
        super().load_checkpoint(epoch, resume_training_state=resume_training_state)


def create_experiment(
    *,
    config: Mapping[str, Any],
    run_directory: str | Path,
    dataset: OrderedDiskTrajDataset,
    components: AMAGOComponents,
    make_train_env: Callable[[], AMAGOEnv] | Iterable[Callable[[], AMAGOEnv]],
    make_validation_env: Callable[[], AMAGOEnv] | Iterable[Callable[[], AMAGOEnv]],
    experiment_type: type[ReasonedExperiment] = ReasonedExperiment,
) -> amago.Experiment:
    """Create an Experiment from explicit official AMAGO components."""
    training = cast(Mapping[str, Any], config["training"])
    environment = cast(Mapping[str, Any], config["environment"])
    tracking = cast(Mapping[str, Any], config.get("tracking", {}))
    run_directory = Path(run_directory)
    force_cpu = config.get("device") == "cpu"
    previous_cpu_override = os.environ.get("ACCELERATE_USE_CPU")
    if force_cpu:
        os.environ["ACCELERATE_USE_CPU"] = "true"
    try:
        experiment = experiment_type(
            run_name=run_directory.name,
            ckpt_base_dir=str(run_directory.parent),
            max_seq_len=int(training["max_sequence_length"]),
            dataset=dataset,
            tstep_encoder_type=components.timestep_encoder,
            traj_encoder_type=components.trajectory_encoder,
            agent_type=components.agent,
            exploration_wrapper_type=components.exploration,
            make_train_env=make_train_env,
            make_val_env=make_validation_env,
            val_timesteps_per_epoch=int(training["validation_timesteps"]),
            parallel_actors=int(environment["parallel_envs"]),
            env_mode="sync",
            sample_actions_train=True,
            sample_actions_val=False,
            log_to_wandb=bool(tracking.get("wandb", False)),
            wandb_project=str(tracking.get("project", "reasoned-icrl")),
            wandb_group_name=(
                None if tracking.get("group") in (None, "") else str(tracking["group"])
            ),
            verbose=bool(training["verbose"]),
            log_interval=int(tracking.get("log_interval", 300)),
            traj_save_len=int(training["trajectory_length"]),
            stagger_traj_file_lengths=False,
            save_trajs_as="npz-compressed",
            dloader_workers=int(training["dataloader_workers"]),
            epochs=int(training["epochs"]),
            start_learning_at_epoch=int(training["start_learning_epoch"]),
            train_timesteps_per_epoch=int(training["timesteps_per_epoch"]),
            train_batches_per_epoch=int(training["batches_per_epoch"]),
            val_interval=int(training["validation_interval"]),
            ckpt_interval=int(training["checkpoint_interval"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            lr_warmup_steps=int(training["warmup_steps"]),
            grad_clip=float(training["gradient_clip"]),
            l2_coeff=float(training["weight_decay"]),
            mixed_precision=str(training["mixed_precision"]),
        )
    finally:
        if force_cpu:
            if previous_cpu_override is None:
                os.environ.pop("ACCELERATE_USE_CPU", None)
            else:
                os.environ["ACCELERATE_USE_CPU"] = previous_cpu_override
    experiment.reasoned_dataset = dataset
    experiment.local_metrics_path = run_directory / TRAINING_METRICS_FILE
    experiment.encoder_architecture_id = components.architecture_id
    protocol = str(config.get("experiment", run_directory.parent.parent.name))
    condition = str(config.get("condition", run_directory.parent.name))
    experiment.wandb_run_name = f"{protocol}/{condition}/{run_directory.name}"
    experiment.wandb_tags = (condition,)
    experiment.wandb_config = {
        "reasoned_icrl/protocol": protocol,
        "reasoned_icrl/condition": condition,
        "reasoned_icrl/seed": config.get("seed"),
        "reasoned_icrl/architecture_id": components.architecture_id,
        "reasoned_icrl/run_directory": str(run_directory),
    }
    experiment.policy_condition = str(config["condition"])
    experiment.reasoned_training_settings = dict(training)
    experiment.reasoned_environment_name = str(environment["name"])
    return experiment


def build_experiment(
    *,
    config: Mapping[str, Any],
    run_directory: str | Path,
    make_train_env: Callable[[], AMAGOEnv] | Iterable[Callable[[], AMAGOEnv]],
    make_validation_env: Callable[[], AMAGOEnv] | Iterable[Callable[[], AMAGOEnv]],
) -> amago.Experiment:
    """Compose baseline model, replay and environment factories."""
    model = dict(cast(Mapping[str, Any], config["model"]))
    uses_history = architecture_uses_history_packet(str(model["architecture_id"]))
    if uses_history:
        factory = (
            make_train_env if callable(make_train_env) else next(iter(make_train_env))
        )
        probe = factory()
        model["public_contract"] = probe.env.get_wrapper_attr("contract")
        probe.close()
    environment = cast(Mapping[str, Any], config["environment"])
    # The environment declares which trunk reads its packets (decision 15).
    model["trunk"] = str(environment.get("encoder", "ff"))
    training = cast(Mapping[str, Any], config["training"])
    components = configure_amago(model, training)
    curriculum = environment.get("curriculum")
    dataset = create_replay_dataset(
        run_directory,
        capacity=int(training["replay_capacity"]),
        full_tasks=uses_history,
        # Only the native fixed-budget meta-task can end on a reset-only step.
        reset_only_terminal=str(environment.get("name")) == "dark_key_to_door",
        relabeler=benchmark_relabeler(config),
        # The one-rule curriculum: replay eligibility follows the learner's
        # measured charged calls through the same warmup/mixed/primary phases.
        curriculum=CurriculumSchedule(
            **dict(curriculum), actors=int(environment["parallel_envs"])
        )
        if isinstance(curriculum, Mapping)
        else None,
    )
    return create_experiment(
        config=config,
        run_directory=run_directory,
        dataset=dataset,
        components=components,
        make_train_env=make_train_env,
        make_validation_env=make_validation_env,
    )


def start_experiment(
    experiment: Any,
    config: Mapping[str, Any],
    *,
    checkpoint_runtime: bool,
) -> str:
    """Validate placement, start AMAGO, and register exact-resume state."""
    requested = str(config.get("device", "auto"))
    actual = str(experiment.DEVICE.type)
    training = cast(Mapping[str, Any], config["training"])
    model = dict(cast(Mapping[str, Any], config["model"]))
    if requested != "auto" and requested != actual:
        raise ContractError(
            f"Requested device {requested!r}, AMAGO selected {actual!r}."
        )
    if actual != "cuda" and str(training.get("mixed_precision", "no")) != "no":
        raise ContractError("Mixed precision requires CUDA with AMAGO 3.4.0.")
    if model.get("attention_backend") == "flash" and actual != "cuda":
        raise ContractError("FlashAttention requires a CUDA device.")
    torch.use_deterministic_algorithms(actual == "cpu")
    if actual == "cpu":
        # Parallel CPU reductions can change low-order bits across a process
        # restart even when every RNG and optimizer value is restored.
        torch.set_num_threads(1)
    experiment.start()
    if actual == "cpu":
        # Make SigmaReparam reductions retain their layout across save/restore.
        for module in experiment.policy.modules():
            for name, buffer in module.named_buffers(recurse=False):
                if not buffer.is_contiguous():
                    setattr(module, name, buffer.contiguous())
    if checkpoint_runtime:
        register_runtime_state(experiment, experiment.reasoned_dataset)
    return actual


def persist_operative_gin(experiment: Any, run_directory: str | Path) -> Path:
    """Retain AMAGO's operative Gin config outside transient checkpoints."""
    source = Path(experiment.ckpt_dir) / "config.txt"
    if not source.is_file():
        raise ContractError("AMAGO did not write its operative Gin configuration.")
    destination = Path(run_directory) / AMAGO_CONFIG_FILE
    architecture_id = str(experiment.encoder_architecture_id)
    destination.write_text(
        f"# Encoder architecture: {architecture_id}\n"
        + source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return destination


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def preflight_config(config: Any) -> dict[str, Any]:
    """Instantiate configured modules without reset/step, replay or output writes."""
    from dataclasses import asdict

    from amago.envs.amago_env import SequenceWrapper

    from reasoned_icrl.runtime.environments import environment_builders

    mapping = config.as_runtime_mapping()
    builders, _ = environment_builders(mapping)
    factory = builders if callable(builders) else builders[0]
    environment = factory()
    try:
        model = dict(mapping["model"])
        if architecture_uses_history_packet(model["architecture_id"]):
            model["public_contract"] = environment.env.get_wrapper_attr("contract")
        model["trunk"] = str(mapping["environment"].get("encoder", "ff"))
        components = configure_amago(model, mapping["training"])
        policy = components.agent(
            obs_space=environment.observation_space,
            rl2_space=SequenceWrapper(environment, save_trajs_to=None).rl2_space["rl2"],
            action_space=environment.action_space,
            max_seq_len=config.training.max_sequence_length,
            tstep_encoder_type=components.timestep_encoder,
            traj_encoder_type=components.trajectory_encoder,
        )
        training = config.training
        return {
            "status": "preflight-only; PENDING MANUAL EXPERIMENTS",
            "run": str(config.run_directory),
            "condition": config.condition,
            "training_seed": config.seed,
            "training_seed_count": 1,
            "model": model,
            "training": asdict(training),
            # A projection, not a measurement: `metrics.json` uses
            # `scalar_training_transitions` for the observed collector total, so
            # preflight must not publish a product under the same key.
            "nominal_collection_product": training.epochs
            * training.timesteps_per_epoch
            * config.environment.parallel_envs,
            "expected_updates_if_replay_ready": (
                training.epochs - training.start_learning_epoch
            )
            * training.batches_per_epoch,
            "epsilon_anneal_vector_calls": training.epsilon_anneal_steps,
            **initialization_audit(policy),
            "initialization": "component-isolated.v1",
            "constructed_cache": cache_measurements(
                policy.traj_encoder.init_hidden_state(
                    config.environment.parallel_envs, next(policy.parameters()).device
                )
            ),
            **({"dat": dat_audit(policy, config)} if config.model.dat else {}),
            **(
                {"state_match": config.model.window_match.to_dict()}
                if config.model.window_match is not None
                else {}
            ),
            "native_objective": {
                name: getattr(policy, name)
                for name in (
                    "critic_loss_weight",
                    "online_coeff",
                    "offline_coeff",
                    "reward_multiplier",
                    "tau",
                    "n_step",
                )
            },
        }
    finally:
        environment.close()


def _state_tensors(hidden: Any) -> list[tuple[str, torch.Tensor]]:
    """Every allocated per-actor rollout tensor of a carrier state, by name.

    This is the audited list: cache slabs of every layer, the source times
    and retained lengths of a rolling cache, the summary's memory and counters,
    AMAGO's key/value caches and sequence index, the GRU's state tensor. A
    summary state's ``initial_memory`` is a detached copy of a parameter shared
    by every actor; it is deliberately not here (see ``shared_state_bytes``),
    so it is neither omitted from the audit nor counted per actor.
    """
    from amago.nets.transformer import TformerHiddenState

    from reasoned_icrl.model.dat_transformer import DATHiddenState
    from reasoned_icrl.model.memo_transformer import MemoHiddenState
    from reasoned_icrl.model.summary_transformer import SummaryHiddenState

    if hidden is None:
        return []
    if isinstance(hidden, torch.Tensor):
        # The GRU carrier's state is one [n_layers, B, d_hidden] tensor.
        return [("hidden", hidden)]
    named: list[tuple[str, torch.Tensor]]
    if isinstance(hidden, DATHiddenState):
        named = [("times", hidden.times), ("lengths", hidden.lengths)]
    elif isinstance(hidden, SummaryHiddenState):
        named = [
            ("memory", hidden.memory),
            ("lengths", hidden.lengths),
            ("segment", hidden.segment),
        ]
    elif isinstance(hidden, MemoHiddenState):
        # The Memo carrier's caches are allocated for the longest task; the
        # live figure per prefix length is reported by ``memo_state_growth``.
        named = [("lengths", hidden.lengths), ("segment", hidden.segment)]
    elif isinstance(hidden, TformerHiddenState):
        return [
            ("key_cache", hidden.key_cache.data),
            ("val_cache", hidden.val_cache.data),
            ("seq_lens", hidden.seq_lens),
        ]
    else:
        raise ContractError(f"Cannot measure cache type {type(hidden).__name__}.")
    for index, layer in enumerate(hidden.layers):
        named += [
            (f"layer_{index}.{name}", value) for name, value in layer.tensors().items()
        ]
    return named


def state_tensor_bytes(hidden: Any) -> dict[str, int]:
    """Allocated bytes of every per-actor state tensor, by name."""
    return {
        name: tensor.numel() * tensor.element_size()
        for name, tensor in _state_tensors(hidden)
    }


def shared_state_bytes(hidden: Any) -> int:
    """Bytes of state a carrier holds once for every actor, not per actor.

    Only the summary carrier has any: the detached ``initial_memory`` it
    writes into a reset row. It is a shadow of the ``memory_init`` parameter
    and is reported here so the per-actor figure cannot be accused of hiding
    it, and so it is never multiplied by the actor count either.
    """
    from reasoned_icrl.model.summary_transformer import SummaryHiddenState

    if isinstance(hidden, SummaryHiddenState):
        memory = hidden.initial_memory
        return int(memory.numel() * memory.element_size())
    return 0


def cache_measurements(hidden: Any) -> dict[str, Any]:
    """Count allocated cache tensors, including integer bookkeeping."""
    tensors = [tensor for _, tensor in _state_tensors(hidden)]
    return {
        "cache_bytes": sum(t.numel() * t.element_size() for t in tensors),
        "cache_float_bytes": sum(
            t.numel() * t.element_size() for t in tensors if t.is_floating_point()
        ),
        "cache_dtypes": sorted({str(t.dtype) for t in tensors}),
    }


def dat_audit(policy: Any, config: Any) -> dict[str, Any]:
    """Measured attention cost, recorded per run rather than assumed.

    Reports the per-layer backend map instead of one label for the model: only
    the selected blocks change, and within them the relational branch is always
    dense even when the content branch uses an optimized backend.
    """
    from reasoned_icrl.experiments.contracts import control_parameter_match
    from reasoned_icrl.model.dat_transformer import (
        content_backend,
        selected_parameter_count,
    )
    from reasoned_icrl.model.trajectory_encoder import (
        SummaryTrajEncoder,
        WindowTrajEncoder,
    )
    from reasoned_icrl.runtime.checkpointing import attention_spec

    carrier = policy.traj_encoder
    backbone = carrier.backbone
    spec = attention_spec(carrier)
    ordinary = str(config.model.attention_backend)
    device = next(backbone.parameters()).device
    # The selected blocks run their own content kernel, chosen by device, so the
    # map reports what executed rather than the setting for the other layers.
    selected_content = content_backend(device)
    # Measured on an allocated single-actor state, tensor by tensor, rather than
    # derived from slot arithmetic: the summary carrier's memory and counters,
    # the window's source times and every layer's slabs are counted as built.
    single = carrier.init_hidden_state(1, device)
    per_row = cache_measurements(single)["cache_bytes"]
    identities: dict[str, str] = {}
    if isinstance(carrier, SummaryTrajEncoder):
        identities["summary_sha256"] = carrier.spec.sha256
    elif isinstance(carrier, WindowTrajEncoder):
        identities["window_sha256"] = carrier.spec.sha256
    return {
        "attention_sha256": spec.sha256,
        "spec": spec.to_dict(),
        **identities,
        "backend_map": {
            str(index): (
                {"content": selected_content, "relational": spec.relational_backend}
                if index in backbone.selected
                else {"content": ordinary}
            )
            for index in range(backbone.n_layers)
        },
        "cache_dtype": spec.cache_dtype,
        "cache_bytes_per_environment": per_row,
        "state_tensor_bytes_per_environment": state_tensor_bytes(single),
        "shared_state_bytes": shared_state_bytes(single),
        "cache_capacity": carrier.capacity,
        "active_attention_parameters": selected_parameter_count(backbone),
        "trajectory_encoder_parameters": parameter_count(carrier),
        **(
            {
                "capacity_match": {
                    "selected_layers": len(backbone.selected),
                    **control_parameter_match(spec),
                }
            }
            if spec.mode == "dual_content"
            else {}
        ),
    }


def initialization_audit(policy: Any) -> dict[str, Any]:
    """Measure actual initial module states and overlapping parameter groups."""
    import hashlib

    groups = {
        "timestep": policy.tstep_encoder,
        "trajectory": policy.traj_encoder,
        "actor": policy.actor,
        "critics": policy.critics,
    }
    if hasattr(policy.tstep_encoder, "spec"):
        groups.update(
            backbone=policy.traj_encoder.backbone,
            token=policy.tstep_encoder.token_encoder,
        )
        # The dual-attention carrier fuses inside its blocks, so it has no
        # separate fusion module. Its selected blocks are hashed on their own so
        # a matched comparison can show the ordinary parts agree and only the
        # replaced attention differs.
        fusion = getattr(policy.traj_encoder, "fusion", None)
        if fusion is not None:
            groups["fusion"] = fusion
        selected: frozenset[int] = getattr(
            policy.traj_encoder.backbone, "selected", frozenset()
        )
        for index in sorted(selected):
            groups[f"dat_attention_{index}"] = policy.traj_encoder.backbone.layers[
                index
            ].attention
        # Every Transformer-family carrier holds the same donor block modules
        # under the same names, so per-block hashes show a bounded cell's
        # ordinary blocks byte-identical to the full-prefix reference's.
        transformer = getattr(
            policy.traj_encoder.backbone, "tformer", policy.traj_encoder.backbone
        )
        blocks = getattr(transformer, "layers", None)
        if blocks is not None:
            for index, block in enumerate(blocks):
                groups[f"block_{index}"] = block
            groups["input_projection"] = transformer.inp
            groups["final_norm"] = transformer.norm
        if policy.tstep_encoder.state_encoder is not None:
            groups["current"] = policy.tstep_encoder.state_encoder
    hashes = {}
    for name, module in groups.items():
        digest = hashlib.sha256()
        for key, value in module.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        hashes[name] = digest.hexdigest()
    return {
        "parameters_total_including_copies": parameter_count(policy),
        "parameters_optimized": sum(p.numel() for p in policy.trainable_params),
        "parameters_by_module": {
            name: parameter_count(module) for name, module in groups.items()
        },
        "initial_state_sha256_by_module": hashes,
    }


def _latency_summary(
    samples: list[float],
) -> tuple[float | None, float | None, float | None]:
    """Mean, 95th percentile and maximum of timed steps; ``None`` when none."""
    if not samples:
        return None, None, None
    values = np.asarray(samples, dtype=np.float64)
    return float(values.mean()), float(np.percentile(values, 95)), float(values.max())


def carrier_cost(
    policy: Any, *, rows: int, device: torch.device, probes: int = 8
) -> dict[str, Any]:
    """Measured per-actor state size, cached decision latency and counted FLOPs.

    A synthetic cached rollout of ``rows`` actors on the live policy: random
    valid packets, one decision per step, timed on the training device.
    ``persistent_state_bytes`` is the allocated rollout state per actor row:
    every cache slot up to the carrier's capacity whether or not it is filled,
    plus the memory tokens and counters; a full-prefix carrier's capacity is
    the task length. ``state_tensor_bytes`` lists that allocation tensor by
    tensor and ``shared_state_bytes`` the summary carrier's one detached copy
    of its initial memory, which is shared by every actor and counted once.
    For the summary carrier the run crosses ``probes`` segment boundaries, and
    the steps that cross one are timed separately (mean, 95th percentile and
    maximum, since a boundary is a spike the decision mean hides); every other
    carrier reports ``None`` for the boundary fields. A window carrier runs
    past its wraparound so the timed decisions include evictions. The first
    decision (the summary read pass, or an empty cache) is excluded from the
    decision statistics.

    FLOPs are counted, not estimated, on a second identical pass under
    ``torch.utils.flop_counter``: the last steady-state decision and the last
    boundary crossing, for the whole ``rows``-actor batch. The counter sees
    matrix-multiply-class operators only (linear layers, batched products
    behind the einsum contractions, fused attention); softmax, normalization
    and elementwise work are not in the figure, which is why it is labelled
    counted rather than total. Nothing here touches an environment.
    """
    from torch.utils.flop_counter import FlopCounterMode

    from reasoned_icrl.model.memo_transformer import MemoHiddenState
    from reasoned_icrl.model.summary_transformer import SummaryHiddenState
    from reasoned_icrl.model.trajectory_encoder import WindowTrajEncoder

    carrier = policy.traj_encoder
    policy.eval()
    single = carrier.init_hidden_state(1, device)
    per_row = cache_measurements(single)["cache_bytes"]
    memo_state = isinstance(single, MemoHiddenState)
    summary_state = isinstance(single, SummaryHiddenState) or memo_state
    window_state = isinstance(carrier, WindowTrajEncoder)
    if probes < 1:
        raise ContractError("The latency probe times at least one boundary.")
    if memo_state:
        # Memo's boundaries grow the cache, so every one of the probed
        # crossings is a different cost; the probe crosses as many as the
        # longest task allows (none on a task shorter than one segment),
        # capped at ``probes``, and reports the growth.
        probes = min(
            probes, int(carrier.spec.summaries_before(carrier.backbone.max_index))
        )
        steps = min(
            probes * int(carrier.spec.segment_length) + 2,
            int(carrier.backbone.max_index) + 1,
        )
    elif summary_state:
        steps = probes * int(carrier.spec.segment_length) + 2
    elif window_state:
        # Past the wraparound, so the timed decisions include evictions and
        # the counted decision reads a full W-slot band.
        steps = int(carrier.capacity) + 8
    else:
        steps = 16
    tstep_dim = int(carrier.tstep_dim)
    generator = torch.Generator(device="cpu").manual_seed(0)
    packets = torch.randn((rows, steps, tstep_dim), generator=generator).to(device)
    packets[..., -1] = 1.0
    times = torch.arange(steps, device=device).view(1, steps, 1).expand(rows, -1, -1)

    def crossing(hidden: Any) -> bool:
        if memo_state:
            due = hidden.lengths == (hidden.held + carrier.spec.segment_length).to(
                hidden.lengths.dtype
            )
            return bool(due.any())
        return bool(
            summary_state and (hidden.lengths == carrier.spec.write_slots.start).any()
        )

    hidden = carrier.init_hidden_state(rows, device)
    decision: list[float] = []
    boundary: list[float] = []
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    with torch.inference_mode():
        for step in range(steps):
            at_boundary = crossing(hidden)
            if sync is not None:
                sync(device)
            started = time.perf_counter()
            _, hidden = carrier(
                packets[:, step : step + 1], times[:, step : step + 1], hidden
            )
            if sync is not None:
                sync(device)
            elapsed = time.perf_counter() - started
            if at_boundary:
                boundary.append(elapsed)
            elif step > 0:
                decision.append(elapsed)
    # The counted pass: the same probe again, under the FLOP counter, so the
    # timed pass above carries none of the counter's dispatch overhead. It runs
    # under no_grad rather than inference mode because the counter's module
    # tracker registers gradient hooks on the parameters it meets, which an
    # inference-mode graph cannot provide.
    with torch.no_grad():
        hidden = carrier.init_hidden_state(rows, device)
        decision_flops: int | None = None
        boundary_flops: int | None = None
        for step in range(steps):
            at_boundary = crossing(hidden)
            counter = FlopCounterMode(display=False)
            with counter:
                _, hidden = carrier(
                    packets[:, step : step + 1], times[:, step : step + 1], hidden
                )
            flops = int(counter.get_total_flops())
            if at_boundary:
                boundary_flops = flops
            elif step > 0:
                decision_flops = flops
    decision_mean, decision_p95, _ = _latency_summary(decision)
    boundary_mean, boundary_p95, boundary_max = _latency_summary(boundary)
    return {
        "persistent_state_bytes": int(per_row),
        "state_tensor_bytes": state_tensor_bytes(single),
        "shared_state_bytes": shared_state_bytes(single),
        "decision_latency_seconds": decision_mean,
        "decision_latency_p95_seconds": decision_p95,
        "boundary_latency_seconds": boundary_mean,
        "boundary_latency_p95_seconds": boundary_p95,
        "boundary_latency_max_seconds": boundary_max,
        "decision_flops_counted": decision_flops,
        "boundary_flops_counted": boundary_flops,
        "flops_method": (
            "torch.utils.flop_counter.FlopCounterMode over one cached decision of "
            "the whole actor batch at the probe's last steady-state step (and its "
            "last boundary crossing for the summary carrier): matrix-multiply-class "
            "operators only; softmax, normalization and elementwise work excluded"
        ),
        "latency_probe": {
            "rows": rows,
            "steps": steps,
            "probes": probes if summary_state else 0,
            "decisions_timed": len(decision),
            "boundaries_timed": len(boundary),
        },
        **({"state_growth": memo_state_growth(carrier)} if memo_state else {}),
    }


def memo_state_growth(carrier: Any) -> dict[str, Any]:
    """The Memo carrier's live state against the prefix length, tensor by tensor.

    The allocation is fixed by the longest task (``persistent_state_bytes``);
    what a frozen policy actually holds grows by ``S`` summary slots at every
    boundary and shrinks by ``L`` record slots (``contracts.memo_live_slots``).
    Reported at the first record, every boundary of the longest task and the
    last record, as bytes of the allocated slabs' filled slots plus the
    counters, so the comparator's "actual state versus prefix length" panel
    reads a measured schedule rather than a nominal slot count.
    """
    from reasoned_icrl.experiments.contracts import memo_live_slots

    spec = carrier.spec
    backbone = carrier.backbone
    single = carrier.init_hidden_state(1, next(backbone.parameters()).device)
    slab_bytes = sum(
        tensor[:, :1].numel() * tensor.element_size()
        for cache in single.layers
        for tensor in cache.tensors().values()
    )
    counter_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (single.lengths, single.segment)
    )
    longest = int(backbone.max_index) + 1
    prefixes = sorted(
        {1, longest}
        | {
            k * spec.segment_length
            for k in range(1, longest // spec.segment_length + 1)
        }
        | {
            k * spec.segment_length + 1
            for k in range(1, longest // spec.segment_length + 1)
            if k * spec.segment_length + 1 <= longest
        }
    )
    return {
        "method": (
            "filled cache slots per layer after P records under the lazy boundary "
            "(q = (P - 1) // L summaries of S slots plus the open segment's P - qL "
            "records) times the allocated bytes per slot, plus the counters"
        ),
        "bytes_per_slot": int(slab_bytes),
        "counter_bytes": int(counter_bytes),
        "capacity_slots": int(carrier.capacity),
        "longest_task_records": longest,
        "live_bytes_by_prefix": {
            str(prefix): int(memo_live_slots(spec, prefix) * slab_bytes + counter_bytes)
            for prefix in prefixes
        },
        "live_slots_by_prefix": {
            str(prefix): int(memo_live_slots(spec, prefix)) for prefix in prefixes
        },
    }
