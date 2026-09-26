"""Serialize baseline caches, validate checkpoints and restore exact AMAGO state."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from amago.envs import SequenceWrapper
from amago.nets.transformer import TformerHiddenState
from torch import nn

from reasoned_icrl.experiments.contracts import (
    ARCHITECTURE_LABELS,
    DAT_WINDOW_ARCHITECTURE_ID,
    WINDOW_ARCHITECTURE_ID,
    ContractError,
    architecture_uses_dat,
    architecture_uses_history_packet,
    architecture_uses_memo,
    architecture_uses_summary,
)
from reasoned_icrl.model.dat_transformer import (
    DAT_HIDDEN_STATE_SCHEMA,
    EMPTY_TIME,
    DATHiddenState,
)
from reasoned_icrl.model.memo_transformer import (
    MEMO_HIDDEN_STATE_SCHEMA,
    MemoHiddenState,
)
from reasoned_icrl.model.summary_transformer import (
    SUMMARY_HIDDEN_STATE_SCHEMA,
    SummaryHiddenState,
)
from reasoned_icrl.model.trajectory_encoder import (
    DATTrajEncoder,
    GRUHistoryTrajEncoder,
    MemoTrajEncoder,
    SummaryTrajEncoder,
    WindowTrajEncoder,
)
from reasoned_icrl.model.window_transformer import (
    DAT_WINDOW_HIDDEN_STATE_SCHEMA,
    WINDOW_HIDDEN_STATE_SCHEMA,
)
from reasoned_icrl.runtime.replay import OrderedDiskTrajDataset

NONE_HIDDEN_STATE_SCHEMA = "amago-hidden-state.none.v1"


TRANSFORMER_HIDDEN_STATE_SCHEMA = "amago-transformer-hidden-state.v1"


GRU_HIDDEN_STATE_SCHEMA = "amago-gru-hidden-state.v1"


def _require_exact_keys(
    state: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(state) != expected:
        raise ContractError(f"{label} checkpoint fields are incompatible.")


def _checkpoint_tensor(state: Mapping[str, object], name: str) -> torch.Tensor:
    value = state.get(name)
    if not isinstance(value, torch.Tensor):
        raise ContractError(f"Checkpoint tensor {name!r} is missing or malformed.")
    return value


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _serialize_transformer_hidden(
    hidden: TformerHiddenState,
) -> dict[str, object]:
    return {
        "schema": TRANSFORMER_HIDDEN_STATE_SCHEMA,
        "key_cache": _cpu_clone(hidden.key_cache.data),
        "val_cache": _cpu_clone(hidden.val_cache.data),
        "seq_lens": _cpu_clone(hidden.seq_lens),
    }


def _restore_transformer_hidden(
    state: Mapping[str, object],
    target: TformerHiddenState,
) -> TformerHiddenState:
    _require_exact_keys(
        state,
        {"schema", "key_cache", "val_cache", "seq_lens"},
        label="AMAGO Transformer hidden-state",
    )
    if state.get("schema") != TRANSFORMER_HIDDEN_STATE_SCHEMA:
        raise ContractError("Unsupported AMAGO Transformer hidden-state version.")
    key_cache = _checkpoint_tensor(state, "key_cache")
    val_cache = _checkpoint_tensor(state, "val_cache")
    seq_lens = _checkpoint_tensor(state, "seq_lens")
    if (
        not key_cache.is_floating_point()
        or val_cache.dtype != key_cache.dtype
        or seq_lens.dtype != torch.int32
        or (seq_lens < 0).any().item()
        or (seq_lens > target.key_cache.max_seq_len).any().item()
    ):
        raise ContractError("AMAGO Transformer hidden-state dtypes are incompatible.")
    tensors = (
        (target.key_cache.data, key_cache, "key cache"),
        (target.val_cache.data, val_cache, "value cache"),
        (target.seq_lens, seq_lens, "sequence index"),
    )
    for destination, source, label in tensors:
        if destination.shape != source.shape:
            raise ContractError(
                f"AMAGO Transformer {label} shape changed across resume."
            )
        destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
    return target


def _hidden_state(hidden: object) -> dict[str, object]:
    """Serialize an AMAGO rollout state: none, Transformer, DAT, summary or GRU.

    Dispatch is on the state object's type, never on a tensor's shape: a bare
    tensor is only ever the GRU carrier's ``[n_layers, B, d_hidden]`` state.
    """
    if hidden is None:
        return {"schema": NONE_HIDDEN_STATE_SCHEMA}
    if isinstance(hidden, TformerHiddenState):
        return _serialize_transformer_hidden(hidden)
    if isinstance(hidden, DATHiddenState):
        return _serialize_dat_hidden(hidden)
    if isinstance(hidden, SummaryHiddenState):
        return _serialize_summary_hidden(hidden)
    if isinstance(hidden, MemoHiddenState):
        return _serialize_memo_hidden(hidden)
    if isinstance(hidden, torch.Tensor):
        return _serialize_gru_hidden(hidden)
    raise ContractError("Unsupported AMAGO rollout hidden-state type.")


def _serialize_memo_hidden(hidden: MemoHiddenState) -> dict[str, object]:
    """Record the counters, every layer cache and the Memo identity."""
    return {
        "schema": MEMO_HIDDEN_STATE_SCHEMA,
        "spec_sha256": hidden.spec_sha256,
        "capacity": hidden.capacity,
        "batch_size": hidden.batch_size,
        "summary_cleared": bool(hidden.summary_cleared),
        "lengths": _cpu_clone(hidden.lengths),
        "segment": _cpu_clone(hidden.segment),
        "layers": [
            {
                "variant": cache.variant,
                "tensors": {
                    name: _cpu_clone(tensor) for name, tensor in cache.tensors().items()
                },
            }
            for cache in hidden.layers
        ],
    }


def _restore_memo_hidden(
    experiment: Any, state: Mapping[str, object]
) -> MemoHiddenState:
    """Rebuild a Memo rollout state on the live carrier, refusing disagreement."""
    _require_exact_keys(
        state,
        {
            "schema",
            "spec_sha256",
            "capacity",
            "batch_size",
            "summary_cleared",
            "lengths",
            "segment",
            "layers",
        },
        label="Memo hidden state",
    )
    carrier = experiment.policy.traj_encoder
    if not isinstance(carrier, MemoTrajEncoder):
        raise ContractError("Checkpoint rollout state does not match the policy.")
    if state["spec_sha256"] != carrier.spec.sha256:
        raise ContractError("Checkpoint Memo identity does not match the model.")
    lengths = _checkpoint_tensor(state, "lengths")
    segment = _checkpoint_tensor(state, "segment")
    if lengths.ndim != 1 or lengths.dtype != torch.int32:
        raise ContractError("Memo lengths must be an int32 [B] tensor.")
    batch = int(lengths.shape[0])
    if segment.shape != (batch,) or segment.dtype != torch.int64:
        raise ContractError("Memo segment counters must be an int64 [B] tensor.")
    if int(cast(int, state["batch_size"])) != batch:
        raise ContractError("Memo hidden-state batch disagrees with its record.")
    if int(cast(int, state["capacity"])) != carrier.capacity:
        raise ContractError("Memo cache capacity does not match the carrier.")
    if not isinstance(state["summary_cleared"], bool):
        raise ContractError("Memo intervention flag must be boolean.")
    if bool((lengths < 0).any()) or bool((lengths > carrier.capacity).any()):
        raise ContractError("Memo filled slots are outside the cache capacity.")
    if bool((segment < 0).any()):
        raise ContractError("Memo boundary counters must be nonnegative.")
    spec = carrier.spec
    held = (
        torch.zeros_like(segment)
        if state["summary_cleared"]
        else (segment * spec.summary_tokens)
    )
    if bool((lengths.long() < held).any()) or bool(
        (lengths.long() > held + spec.segment_length).any()
    ):
        raise ContractError(
            "Memo filled slots disagree with the boundaries crossed: a row holds "
            "its summaries plus at most one open segment."
        )
    restored: MemoHiddenState = carrier.init_hidden_state(batch, experiment.DEVICE)
    layers = cast(Sequence[Mapping[str, Any]], state["layers"])
    if len(layers) != restored.n_layers:
        raise ContractError("Memo hidden state has the wrong number of layers.")
    for record, cache in zip(layers, restored.layers, strict=True):
        if record["variant"] != cache.variant:
            raise ContractError("Memo cached layer variant does not match.")
        stored = cache.tensors()
        if set(record["tensors"]) != set(stored):
            raise ContractError("Memo cached layer tensors do not match.")
        for name, tensor in stored.items():
            source = record["tensors"][name]
            if source.shape != tensor.shape or source.dtype != tensor.dtype:
                raise ContractError(f"Memo cached {name!r} changed shape or dtype.")
            for row in range(batch):
                filled = int(lengths[row].item())
                if filled and not bool(torch.isfinite(source[row, :filled]).all()):
                    raise ContractError(f"Filled Memo {name!r} is not finite.")
            tensor.copy_(source.to(tensor.device))
    restored.lengths.copy_(lengths.to(restored.lengths.device))
    restored.segment.copy_(segment.to(restored.segment.device))
    restored.summary_cleared = bool(state["summary_cleared"])
    return restored


def _serialize_summary_hidden(hidden: SummaryHiddenState) -> dict[str, object]:
    """Record the memory, counters, every layer cache and the summary identity."""
    return {
        "schema": SUMMARY_HIDDEN_STATE_SCHEMA,
        "spec_sha256": hidden.spec_sha256,
        "capacity": hidden.capacity,
        "batch_size": hidden.batch_size,
        "summary_cleared": bool(hidden.summary_cleared),
        "memory": _cpu_clone(hidden.memory),
        "lengths": _cpu_clone(hidden.lengths),
        "segment": _cpu_clone(hidden.segment),
        "layers": [
            {
                "variant": cache.variant,
                "tensors": {
                    name: _cpu_clone(tensor) for name, tensor in cache.tensors().items()
                },
            }
            for cache in hidden.layers
        ],
    }


def _restore_summary_hidden(
    experiment: Any, state: Mapping[str, object]
) -> SummaryHiddenState:
    """Rebuild a summary rollout state on the live carrier, refusing disagreement."""
    _require_exact_keys(
        state,
        {
            "schema",
            "spec_sha256",
            "capacity",
            "batch_size",
            "summary_cleared",
            "memory",
            "lengths",
            "segment",
            "layers",
        },
        label="Summary hidden state",
    )
    carrier = experiment.policy.traj_encoder
    if not isinstance(carrier, SummaryTrajEncoder):
        raise ContractError("Checkpoint rollout state does not match the policy.")
    if state["spec_sha256"] != carrier.spec.sha256:
        raise ContractError("Checkpoint summary identity does not match the model.")
    memory = _checkpoint_tensor(state, "memory")
    lengths = _checkpoint_tensor(state, "lengths")
    segment = _checkpoint_tensor(state, "segment")
    if memory.ndim != 3 or memory.dtype != torch.float32:
        raise ContractError("Summary memory must be a float32 [B, M, d] tensor.")
    batch = int(memory.shape[0])
    if (
        lengths.shape != (batch,)
        or lengths.dtype != torch.int32
        or segment.shape != (batch,)
        or segment.dtype != torch.int64
    ):
        raise ContractError("Summary counters must be int32/int64 [B] tensors.")
    if int(cast(int, state["batch_size"])) != batch:
        raise ContractError("Summary hidden-state batch disagrees with its record.")
    if int(cast(int, state["capacity"])) != carrier.capacity:
        raise ContractError("Summary cache capacity does not match the carrier.")
    if not isinstance(state["summary_cleared"], bool):
        raise ContractError("Summary intervention flag must be boolean.")
    if bool((lengths < 0).any()) or bool((lengths > carrier.capacity).any()):
        raise ContractError("Summary filled slots are outside the segment capacity.")
    if bool((segment < 0).any()):
        raise ContractError("Summary boundary counters must be nonnegative.")
    if not bool(torch.isfinite(memory).all()):
        raise ContractError("Summary memory is not finite.")
    restored: SummaryHiddenState = carrier.init_hidden_state(batch, experiment.DEVICE)
    if restored.memory.shape != memory.shape:
        raise ContractError("Summary memory shape does not match the carrier.")
    layers = cast(Sequence[Mapping[str, Any]], state["layers"])
    if len(layers) != restored.n_layers:
        raise ContractError("Summary hidden state has the wrong number of layers.")
    for record, cache in zip(layers, restored.layers, strict=True):
        if record["variant"] != cache.variant:
            raise ContractError("Summary cached layer variant does not match.")
        stored = cache.tensors()
        if set(record["tensors"]) != set(stored):
            raise ContractError("Summary cached layer tensors do not match.")
        for name, tensor in stored.items():
            source = record["tensors"][name]
            if source.shape != tensor.shape or source.dtype != tensor.dtype:
                raise ContractError(f"Summary cached {name!r} changed shape or dtype.")
            for row in range(batch):
                filled = int(lengths[row].item())
                if filled and not bool(torch.isfinite(source[row, :filled]).all()):
                    raise ContractError(f"Filled summary {name!r} is not finite.")
            tensor.copy_(source.to(tensor.device))
    restored.memory.copy_(memory.to(restored.memory.device))
    restored.lengths.copy_(lengths.to(restored.lengths.device))
    restored.segment.copy_(segment.to(restored.segment.device))
    restored.summary_cleared = bool(state["summary_cleared"])
    return restored


def _serialize_gru_hidden(hidden: torch.Tensor) -> dict[str, object]:
    """Record the GRU state tensor with its geometry."""
    if hidden.ndim != 3:
        raise ContractError("GRU hidden state must be [layers, B, d_hidden].")
    layers, batch, width = hidden.shape
    return {
        "schema": GRU_HIDDEN_STATE_SCHEMA,
        "layers": int(layers),
        "batch_size": int(batch),
        "d_hidden": int(width),
        "hidden": _cpu_clone(hidden),
    }


def _restore_gru_hidden(experiment: Any, state: Mapping[str, object]) -> torch.Tensor:
    """Rebuild the GRU state on the live carrier, rejecting any disagreement."""
    _require_exact_keys(
        state,
        {"schema", "layers", "batch_size", "d_hidden", "hidden"},
        label="GRU hidden state",
    )
    carrier = experiment.policy.traj_encoder
    if not isinstance(carrier, GRUHistoryTrajEncoder):
        raise ContractError("Checkpoint rollout state does not match the policy.")
    hidden = _checkpoint_tensor(state, "hidden")
    if hidden.ndim != 3 or hidden.dtype != torch.float32:
        raise ContractError("GRU hidden state must be a float32 [layers, B, d_hidden].")
    layers, batch, width = (int(value) for value in hidden.shape)
    recorded = tuple(
        int(cast(int, state[name])) for name in ("layers", "batch_size", "d_hidden")
    )
    if recorded != (layers, batch, width):
        raise ContractError("GRU hidden-state shape disagrees with its record.")
    if layers != carrier.n_layers or width != carrier.emb_dim:
        raise ContractError("GRU hidden state does not match the carrier.")
    if not bool(torch.isfinite(hidden).all()):
        raise ContractError("GRU hidden state is not finite.")
    restored: torch.Tensor = carrier.init_hidden_state(batch, experiment.DEVICE)
    restored.copy_(hidden.to(restored.device))
    return restored


_ROLLING_CACHE_SCHEMAS: dict[str, tuple[tuple[str, ...], str, type[nn.Module]]] = {
    DAT_HIDDEN_STATE_SCHEMA: (
        ("attention_sha256",),
        "DAT hidden state",
        DATTrajEncoder,
    ),
    WINDOW_HIDDEN_STATE_SCHEMA: (
        ("window_sha256",),
        "Window hidden state",
        WindowTrajEncoder,
    ),
    DAT_WINDOW_HIDDEN_STATE_SCHEMA: (
        ("window_sha256", "attention_sha256"),
        "DAT window hidden state",
        WindowTrajEncoder,
    ),
}
"""The carriers that hold a rolling ``DATHiddenState``: the identity keys each
schema records, its label, and the carrier it restores onto. The dual-attention
window records both its window and its attention identity and restores only
where both agree."""


def _hidden_identities(hidden: DATHiddenState) -> dict[str, str]:
    """The identities a rolling-cache state records, keyed as its schema names them."""
    if hidden.schema == DAT_HIDDEN_STATE_SCHEMA:
        return {"attention_sha256": hidden.attention_sha256}
    if hidden.schema == WINDOW_HIDDEN_STATE_SCHEMA:
        return {"window_sha256": hidden.attention_sha256}
    if hidden.schema == DAT_WINDOW_HIDDEN_STATE_SCHEMA:
        if hidden.window_sha256 is None:
            raise ContractError("A DAT window state needs its window identity.")
        return {
            "window_sha256": hidden.window_sha256,
            "attention_sha256": hidden.attention_sha256,
        }
    raise ContractError(f"Unsupported rolling-cache schema: {hidden.schema!r}.")


def _carrier_identities(schema: str, carrier: Any) -> dict[str, str]:
    """The identities the live carrier would record under ``schema``.

    Raises when the carrier is not the kind the schema restores onto: an
    ordinary window state never lands on a dual-attention window and vice
    versa, whatever the individual hashes say.
    """
    _, _, carrier_type = _ROLLING_CACHE_SCHEMAS[schema]
    if not isinstance(carrier, carrier_type):
        raise ContractError("Checkpoint rollout cache does not match the policy.")
    live: Any = carrier
    attention = getattr(live, "dat", None)
    if schema == DAT_HIDDEN_STATE_SCHEMA:
        return {"attention_sha256": str(live.spec.sha256)}
    if schema == WINDOW_HIDDEN_STATE_SCHEMA:
        if attention is not None:
            raise ContractError("Checkpoint rollout cache does not match the policy.")
        return {"window_sha256": str(live.spec.sha256)}
    if attention is None:
        raise ContractError("Checkpoint rollout cache does not match the policy.")
    return {
        "window_sha256": str(live.spec.sha256),
        "attention_sha256": str(attention.sha256),
    }


def _serialize_dat_hidden(hidden: DATHiddenState) -> dict[str, object]:
    """Record every layer cache, the shared times, lengths and the carrier identities.

    Serves the DAT and the two window carriers, which share the rolling-cache
    state class; the state's schema says which, and names the identity keys.
    """
    if hidden.schema not in _ROLLING_CACHE_SCHEMAS:
        raise ContractError(f"Unsupported rolling-cache schema: {hidden.schema!r}.")
    return {
        "schema": hidden.schema,
        **_hidden_identities(hidden),
        "capacity": hidden.capacity,
        "batch_size": hidden.batch_size,
        "lengths": hidden.lengths.detach().cpu().clone(),
        "times": hidden.times.detach().cpu().clone(),
        "layers": [
            {
                "variant": cache.variant,
                "tensors": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in cache.tensors().items()
                },
            }
            for cache in hidden.layers
        ],
    }


def _restore_dat_hidden(
    experiment: Any,
    state: Mapping[str, object],
    *,
    schema: str = DAT_HIDDEN_STATE_SCHEMA,
) -> DATHiddenState:
    """Rebuild a DAT or window rollout cache, rejecting any semantic disagreement.

    Dispatch is on the carrier: the DAT schema restores onto ``DATTrajEncoder``
    and the two window schemas onto ``WindowTrajEncoder`` with or without its
    dual-attention spec, each against every identity the schema records.
    """
    identities, label, _ = _ROLLING_CACHE_SCHEMAS[schema]
    _require_exact_keys(
        state,
        {"schema", *identities, "capacity", "batch_size", "lengths", "times", "layers"},
        label=label,
    )
    carrier: Any = experiment.policy.traj_encoder
    expected = _carrier_identities(schema, carrier)
    for key in identities:
        if state[key] != expected[key]:
            raise ContractError(
                "Checkpoint attention identity does not match the model."
                if key == "attention_sha256"
                else "Checkpoint window identity does not match the model."
            )
    lengths = _checkpoint_tensor(state, "lengths")
    times = _checkpoint_tensor(state, "times")
    if lengths.ndim != 1 or times.ndim != 2:
        raise ContractError("DAT lengths must be [B] and times [B, C].")
    batch, capacity = times.shape
    if (
        int(cast(int, state["batch_size"])) != batch
        or int(cast(int, state["capacity"])) != capacity
    ):
        raise ContractError("DAT hidden-state shape disagrees with its record.")
    if capacity != carrier.capacity:
        raise ContractError("DAT cache capacity does not match the carrier.")
    if bool((lengths < 0).any()) or bool((lengths >= capacity).any()):
        raise ContractError("DAT retained lengths are outside the cache capacity.")
    restored: DATHiddenState = carrier.init_hidden_state(batch, experiment.DEVICE)
    layers = cast(Sequence[Mapping[str, Any]], state["layers"])
    if len(layers) != restored.n_layers:
        raise ContractError("DAT hidden state has the wrong number of layers.")
    for record, cache in zip(layers, restored.layers, strict=True):
        if record["variant"] != cache.variant:
            raise ContractError("DAT cached layer variant does not match the model.")
        stored = cache.tensors()
        if set(record["tensors"]) != set(stored):
            raise ContractError("DAT cached layer tensors do not match the model.")
        for name, tensor in stored.items():
            source = record["tensors"][name]
            if source.shape != tensor.shape or source.dtype != tensor.dtype:
                raise ContractError(f"DAT cached {name!r} changed shape or dtype.")
            tensor.copy_(source.to(tensor.device))
    restored.lengths.copy_(lengths.to(restored.lengths.device))
    restored.times.copy_(times.to(restored.times.device))
    _validate_dat_positions(restored)
    return restored


def _validate_dat_positions(hidden: DATHiddenState) -> None:
    """Retained slots must be finite, strictly increasing and free of task mixing."""
    for row in range(hidden.batch_size):
        length = int(hidden.lengths[row].item())
        used = hidden.times[row, :length]
        if length and bool((used == EMPTY_TIME).any()):
            raise ContractError("A retained DAT slot has no source position.")
        if length > 1 and not bool((used[1:] > used[:-1]).all()):
            raise ContractError("DAT source positions must strictly increase.")
        if bool((hidden.times[row, length:] != EMPTY_TIME).any()):
            raise ContractError("DAT holds a source beyond its retained length.")
        for cache in hidden.layers:
            for name, tensor in cache.tensors().items():
                if length and not bool(torch.isfinite(tensor[row, :length]).all()):
                    raise ContractError(f"Retained DAT {name!r} is not finite.")


def _restore_hidden_state(experiment: Any, state: Mapping[str, object]) -> object:
    schema = state.get("schema")
    if not isinstance(schema, str):
        raise ContractError("AMAGO hidden-state checkpoint lacks a versioned schema.")
    if schema == NONE_HIDDEN_STATE_SCHEMA:
        _require_exact_keys(state, {"schema"}, label="No-hidden-state")
        return None
    if schema in _ROLLING_CACHE_SCHEMAS:
        return _restore_dat_hidden(experiment, state, schema=schema)
    if schema == GRU_HIDDEN_STATE_SCHEMA:
        return _restore_gru_hidden(experiment, state)
    if schema == SUMMARY_HIDDEN_STATE_SCHEMA:
        return _restore_summary_hidden(experiment, state)
    if schema == MEMO_HIDDEN_STATE_SCHEMA:
        return _restore_memo_hidden(experiment, state)
    if schema != TRANSFORMER_HIDDEN_STATE_SCHEMA:
        raise ContractError(f"Unsupported AMAGO hidden-state schema: {schema!r}.")
    seq_lens = _checkpoint_tensor(state, "seq_lens")
    if seq_lens.ndim != 1:
        raise ContractError("AMAGO Transformer sequence index must have shape [B].")
    restored = experiment.policy.traj_encoder.init_hidden_state(
        int(seq_lens.shape[0]), experiment.DEVICE
    )
    if not isinstance(restored, TformerHiddenState):
        raise ContractError("Checkpoint rollout cache does not match the policy.")
    return _restore_transformer_hidden(state, restored)


RUNTIME_STATE_SCHEMA = "amago-runtime-state.v4"


RUNTIME_CONTRACT_SCHEMA = "reasoned-icrl-runtime-contract.v4"


def _sequence_wrappers(environment: Any, mode: str) -> tuple[SequenceWrapper, ...]:
    if mode == "already_vectorized":
        values = (getattr(environment, "env", None),)
    elif mode == "sync":
        values = tuple(getattr(environment, "envs", ()))
    else:
        raise ContractError(
            "Exact resume supports only sync and already_vectorized AMAGO modes."
        )
    if not values or not all(isinstance(value, SequenceWrapper) for value in values):
        raise ContractError("AMAGO's environment wrapper layout changed from v3.4.0.")
    return cast(tuple[SequenceWrapper, ...], values)


def _sequence_state(sequence: SequenceWrapper) -> dict[str, object]:
    initialized = hasattr(sequence, "_current_timestep")
    if not initialized:
        return {"initialized": False}
    if sequence.finished_trajs:
        raise ContractError(
            "AMAGO checkpoints require completed trajectories to be flushed first."
        )
    reader = getattr(sequence.env, "state_dict", None)
    if not callable(reader):
        raise ContractError("The AMAGO environment stack is not restorable.")
    return {
        "initialized": True,
        "environment": deepcopy(reader()),
        "current_timestep": deepcopy(sequence._current_timestep),
        "active_trajs": deepcopy(sequence.active_trajs),
        "since_last_save": list(sequence.since_last_save),
        "save_this_time": list(sequence.save_this_time),
        "total_return": np.asarray(sequence.total_return).copy(),
        "total_frames": int(sequence._total_frames),
        "total_frames_by_env_name": dict(sequence._total_frames_by_env_name),
        "return_history": deepcopy(sequence.return_history.data),
        "special_history": deepcopy(sequence.special_history.data),
    }


def _restore_sequence(sequence: SequenceWrapper, state: Mapping[str, object]) -> None:
    initialized = bool(state.get("initialized", False))
    currently_initialized = hasattr(sequence, "_current_timestep")
    if not initialized:
        if currently_initialized:
            raise ContractError("Checkpoint and validation-environment state disagree.")
        return
    loader = getattr(sequence.env, "load_state_dict", None)
    environment = state.get("environment")
    if not callable(loader) or not isinstance(environment, Mapping):
        raise ContractError("The AMAGO environment stack is not restorable.")
    loader(deepcopy(environment))
    sequence._current_timestep = deepcopy(state["current_timestep"])
    sequence.active_trajs = deepcopy(state["active_trajs"])
    sequence.since_last_save = [
        int(value) for value in cast(Sequence[Any], state["since_last_save"])
    ]
    sequence.save_this_time = [
        None if value is None else int(value)
        for value in cast(Sequence[Any], state["save_this_time"])
    ]
    sequence.total_return = np.asarray(state["total_return"], dtype=np.float64).copy()
    sequence._total_frames = int(cast(int, state["total_frames"]))
    sequence._total_frames_by_env_name = defaultdict(
        int,
        {
            str(key): int(value)
            for key, value in cast(
                Mapping[str, Any], state["total_frames_by_env_name"]
            ).items()
        },
    )
    sequence.finished_trajs = []
    sequence.return_history.data = deepcopy(state["return_history"])
    sequence.special_history.data = deepcopy(state["special_history"])


def _runtime_contract(experiment: Any) -> dict[str, object]:
    contract: dict[str, object] = {
        "schema": RUNTIME_CONTRACT_SCHEMA,
        "architecture_id": str(experiment.encoder_architecture_id),
        "learner_contract": getattr(experiment, "learner_contract", "unqualified"),
        "training_settings": getattr(experiment, "reasoned_training_settings", {}),
    }
    if architecture_uses_history_packet(str(experiment.encoder_architecture_id)):
        contract.update(
            {
                "condition": experiment.policy_condition,
                "packet_sha256": experiment.policy.tstep_encoder.spec.sha256,
            }
        )
    carrier = experiment.policy.traj_encoder
    if architecture_uses_summary(str(experiment.encoder_architecture_id)):
        contract["summary_sha256"] = carrier.spec.sha256
    if architecture_uses_memo(str(experiment.encoder_architecture_id)):
        contract["memo_sha256"] = carrier.spec.sha256
    if str(experiment.encoder_architecture_id) in (
        WINDOW_ARCHITECTURE_ID,
        DAT_WINDOW_ARCHITECTURE_ID,
    ):
        contract["window_sha256"] = carrier.spec.sha256
    if architecture_uses_dat(str(experiment.encoder_architecture_id)):
        # A matching packet hash cannot identify the attention: the timestep
        # encoder is the same for the ordinary and dual-attention conditions.
        contract["attention_sha256"] = attention_spec(carrier).sha256
    return contract


def attention_spec(carrier: Any) -> Any:
    """The carrier's dual-attention identity: ``dat`` on the summary carrier."""
    spec = getattr(carrier, "dat", None)
    return carrier.spec if spec is None else spec


def _validate_runtime_contract(experiment: Any, contract: Mapping[str, object]) -> None:
    if contract.get("schema") != RUNTIME_CONTRACT_SCHEMA:
        raise ContractError("Unsupported AMAGO runtime model-contract version.")
    expected = _runtime_contract(experiment)
    _require_exact_keys(
        contract,
        set(expected),
        label="AMAGO runtime model contract",
    )
    if contract != expected:
        raise ContractError("Runtime checkpoint architecture is incompatible.")


class AMAGORuntimeState:
    """Accelerate checkpoint adapter for live rollout and replay state."""

    def __init__(self, experiment: Any, dataset: OrderedDiskTrajDataset) -> None:
        self.experiment = experiment
        self.dataset = dataset
        self.allow_missing_replay = False
        self.mode = str(experiment.env_mode)
        if self.mode not in ("sync", "already_vectorized"):
            raise ContractError(
                "Exact resume supports only sync and already_vectorized AMAGO modes."
            )

    def state_dict(self) -> dict[str, object]:
        train = _sequence_wrappers(self.experiment.train_envs, self.mode)
        validation = _sequence_wrappers(self.experiment.val_envs, self.mode)
        return {
            "schema": RUNTIME_STATE_SCHEMA,
            "model_contract": _runtime_contract(self.experiment),
            "mode": self.mode,
            "train_environments": [_sequence_state(value) for value in train],
            "validation_environments": [_sequence_state(value) for value in validation],
            "hidden_state": _hidden_state(
                self.experiment.hidden_state,
            ),
            "grad_update_counter": int(self.experiment.grad_update_counter),
            "replay": self.dataset.state_dict(),
        }

    def validate_state_dict(self, state: Mapping[str, object]) -> object:
        """Validate model identity and hidden tensors without mutating live state."""
        schema = state.get("schema")
        if schema != RUNTIME_STATE_SCHEMA:
            raise ContractError("Unsupported AMAGO runtime checkpoint state.")
        expected_fields = {
            "schema",
            "model_contract",
            "mode",
            "train_environments",
            "validation_environments",
            "hidden_state",
            "grad_update_counter",
            "replay",
        }
        _require_exact_keys(
            state,
            expected_fields,
            label="AMAGO runtime-state",
        )
        if state.get("mode") != self.mode:
            raise ContractError("AMAGO environment mode changed across resume.")
        model_contract = state.get("model_contract")
        if not isinstance(model_contract, Mapping):
            raise ContractError("AMAGO runtime checkpoint lacks its model contract.")
        _validate_runtime_contract(self.experiment, model_contract)
        train = _sequence_wrappers(self.experiment.train_envs, self.mode)
        validation = _sequence_wrappers(self.experiment.val_envs, self.mode)
        raw_train_value = state.get("train_environments")
        raw_validation_value = state.get("validation_environments")
        if (
            not isinstance(raw_train_value, Sequence)
            or isinstance(raw_train_value, str)
            or not all(isinstance(value, Mapping) for value in raw_train_value)
            or not isinstance(raw_validation_value, Sequence)
            or isinstance(raw_validation_value, str)
            or not all(isinstance(value, Mapping) for value in raw_validation_value)
        ):
            raise ContractError("AMAGO runtime environment state is malformed.")
        raw_train = cast(Sequence[Mapping[str, object]], raw_train_value)
        raw_validation = cast(Sequence[Mapping[str, object]], raw_validation_value)
        if len(train) != len(raw_train) or len(validation) != len(raw_validation):
            raise ContractError("AMAGO actor count changed across resume.")
        hidden = state.get("hidden_state")
        replay = state.get("replay")
        if not isinstance(hidden, Mapping) or not isinstance(replay, Mapping):
            raise ContractError("AMAGO runtime checkpoint is malformed.")
        counter = state.get("grad_update_counter")
        if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
            raise ContractError("AMAGO gradient-update counter is malformed.")
        return _restore_hidden_state(
            self.experiment,
            hidden,
        )

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        restored_hidden = self.validate_state_dict(state)
        train = _sequence_wrappers(self.experiment.train_envs, self.mode)
        validation = _sequence_wrappers(self.experiment.val_envs, self.mode)
        raw_train = cast(Sequence[Mapping[str, object]], state["train_environments"])
        raw_validation = cast(
            Sequence[Mapping[str, object]], state["validation_environments"]
        )
        for wrapper, value in zip(train, raw_train, strict=True):
            _restore_sequence(wrapper, value)
        for wrapper, value in zip(validation, raw_validation, strict=True):
            _restore_sequence(wrapper, value)
        self.experiment.hidden_state = restored_hidden
        self.experiment.grad_update_counter = int(
            cast(int, state["grad_update_counter"])
        )
        replay = cast(Mapping[str, object], state["replay"])
        self.dataset.load_state_dict(replay, allow_missing=self.allow_missing_replay)


def register_runtime_state(
    experiment: Any, dataset: OrderedDiskTrajDataset
) -> AMAGORuntimeState:
    """Register runtime state after ``Experiment.start()`` and before loading."""
    adapter = AMAGORuntimeState(experiment, dataset)
    experiment.accelerator.register_for_checkpointing(adapter)
    experiment.reasoned_runtime_state = adapter
    return adapter


def validate_checkpoint_architecture(
    state: Mapping[str, Any],
    expected: str,
    *,
    expected_state: Mapping[str, torch.Tensor],
) -> None:
    """Reject architecture/spec/shape mismatches before loading any tensors."""
    if expected not in ARCHITECTURE_LABELS:
        raise ContractError("Unsupported checkpoint architecture.")
    if set(state) != set(expected_state):
        raise ContractError("Checkpoint parameter and buffer keys are incompatible.")
    for name, target in expected_state.items():
        source = state.get(name)
        if name.endswith("protocol_identity") and (
            not isinstance(source, torch.Tensor)
            or not torch.equal(source.cpu(), target.cpu())
        ):
            raise ContractError(
                "History packet protocol, fields, or condition changed."
            )
        if not isinstance(source, torch.Tensor) or source.shape != target.shape:
            raise ContractError(
                f"Checkpoint tensor {name!r} shape is incompatible with the model."
            )


def policy_checkpoint(
    state: Mapping[str, torch.Tensor], *, condition: str, architecture_id: str
) -> dict[str, object]:
    """Versioned policy artifact, independent of Accelerate training-state files."""
    return {
        "schema": "reasoned-icrl-policy.v2",
        "config_version": "0.8.0",
        "condition": condition,
        "architecture_id": architecture_id,
        "state_dict": dict(state),
    }


def read_policy_checkpoint(
    payload: object, *, condition: str, architecture_id: str
) -> Mapping[str, torch.Tensor]:
    """Reject older raw state dictionaries before loading any model tensor."""
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {"schema", "config_version", "condition", "architecture_id", "state_dict"}
        or payload.get("schema") != "reasoned-icrl-policy.v2"
        or payload.get("config_version") != "0.8.0"
        or payload.get("condition") != condition
        or payload.get("architecture_id") != architecture_id
    ):
        raise ContractError("Unsupported or incompatible policy checkpoint contract.")
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ContractError("Checkpoint is not a tensor mapping.")
    return cast(Mapping[str, torch.Tensor], state)


def snapshot_reproduction_material(
    checkpoint_root: Path,
    dataset: Any,
    telemetry: Path,
) -> None:
    """Retain checkpoint replay against FIFO eviction and record a log boundary.

    Published last: absence of this marker means an incomplete retained checkpoint.
    Immutable replay bytes use hard links when possible, portable copies otherwise.
    """
    import json
    import os
    import shutil

    rows = dataset.state_dict()["ordered"]
    for row in rows:
        source = (
            Path(dataset.fifo_path if row["area"] == "fifo" else dataset.protected_path)
            / row["name"]
        )
        target = checkpoint_root / "replay_snapshot" / row["area"] / row["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
    payload = {
        "schema": "darkroom-retained-checkpoint.v1",
        "replay": rows,
        "telemetry_bytes": telemetry.stat().st_size if telemetry.is_file() else 0,
    }
    temporary = checkpoint_root / "reproduction-complete.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(checkpoint_root / "reproduction-complete.json")


def restore_reproduction_material(
    checkpoint_root: Path,
    dataset: Any,
    telemetry: Path,
) -> None:
    """Recover evicted replay and archive abandoned telemetry after interruption."""
    import json
    import shutil

    marker = checkpoint_root / "reproduction-complete.json"
    if not marker.is_file():
        raise ContractError("Retained reproduction checkpoint is incomplete.")
    payload = json.loads(marker.read_text())
    if payload.get("schema") != "darkroom-retained-checkpoint.v1":
        raise ContractError("Retained reproduction checkpoint schema changed.")
    for row in payload["replay"]:
        area, name = row["area"], row["name"]
        if area not in ("fifo", "protected") or Path(name).name != name:
            raise ContractError("Unsafe retained replay path.")
        source = checkpoint_root / "replay_snapshot" / area / name
        target = (
            Path(dataset.fifo_path if area == "fifo" else dataset.protected_path) / name
        )
        if not source.is_file():
            raise ContractError("Retained replay snapshot is missing.")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if telemetry.is_file():
        boundary = int(payload["telemetry_bytes"])
        content = telemetry.read_bytes()
        if len(content) < boundary:
            raise ContractError("Retained checkpoint telemetry prefix is missing.")
        if len(content) > boundary:
            archive = telemetry.parent / "interrupted-telemetry"
            archive.mkdir(exist_ok=True)
            index = len(list(archive.iterdir()))
            (archive / f"suffix-{index:04d}.jsonl").write_bytes(content[boundary:])
            telemetry.write_bytes(content[:boundary])
