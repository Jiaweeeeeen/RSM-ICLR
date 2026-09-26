"""Project-owned trajectory encoders and the history packet view.

The timestep encoder hands over a packet of ``[token | state | valid]``. Only
the ``token`` columns enter the sequence model (a causal Transformer, AMAGO's
GRU for the recurrent baseline, or the segment/summary carrier); the ``state``
columns, when the bypass is on, travel around it and are fused with the
Transformer output just before the actor and critic see anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gin
import numpy as np
import torch
from amago.nets.traj_encoders import GRUTrajEncoder, TformerTrajEncoder, TrajEncoder
from amago.nets.transformer import TformerHiddenState
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from reasoned_icrl.experiments.contracts import (
    ContractError,
    DATSpec,
    MemoSpec,
    SummarySpec,
    WindowSpec,
)
from reasoned_icrl.model.dat_transformer import DATHiddenState, DATTransformer
from reasoned_icrl.model.memo_transformer import MemoHiddenState, MemoTransformer
from reasoned_icrl.model.summary_transformer import (
    SummaryHiddenState,
    SummaryTransformer,
)
from reasoned_icrl.model.utils import module_seed
from reasoned_icrl.model.window_transformer import WindowTransformer

TOKEN_DIM = 64
"""Default token width; also the bypassed state width when the bypass is on."""


@dataclass(frozen=True, slots=True)
class HistoryPacket:
    """A decoded packet: memory ``[B,T,token]``, state ``[B,T,s]``, valid ``[B,T,1]``.

    ``state`` has width zero when the condition runs without the bypass.
    """

    memory: torch.Tensor
    state: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def unpack(
        cls, packet: torch.Tensor, *, token_dim: int, state_dim: int
    ) -> HistoryPacket:
        if packet.ndim != 3 or packet.shape[-1] != token_dim + state_dim + 1:
            raise ContractError("Invalid history trajectory packet shape.")
        valid = packet[..., -1:] == 1
        clean = torch.where(valid, packet, 0.0)
        return cls(
            clean[..., :token_dim], clean[..., token_dim : token_dim + state_dim], valid
        )


@gin.configurable
class HistoryTrajEncoder(TrajEncoder):  # type: ignore[misc]
    """Baseline carrier: an AMAGO Transformer over the packet's token columns.

    Without the bypass the Transformer output reaches the heads unmodified;
    with it, the current-state columns are fused in after the Transformer.
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        bypass: bool = True,
        token_dim: int = TOKEN_DIM,
        d_model: int = 256,
    ) -> None:
        super().__init__(tstep_dim, max_seq_len)
        self.token_dim = token_dim
        self.state_dim = token_dim if bypass else 0
        if tstep_dim != self.token_dim + self.state_dim + 1:
            raise ContractError(
                "History packet width does not match the declared token/bypass split."
            )
        self.backbone = TformerTrajEncoder(
            tstep_dim=token_dim, max_seq_len=max_seq_len + 1
        )
        if self.backbone.emb_dim != d_model:
            raise ContractError("History backbone/output width mismatch.")
        self.bypass = bypass
        self.output_dim = d_model
        self.fusion: nn.Module = (
            nn.Linear(d_model + token_dim, d_model) if bypass else nn.Identity()
        )

    @property
    def emb_dim(self) -> int:
        return self.output_dim

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> TformerHiddenState:
        return self.backbone.init_hidden_state(batch_size, device)

    def reset_hidden_state(
        self, hidden_state: TformerHiddenState | None, dones: np.ndarray
    ) -> TformerHiddenState | None:
        if hidden_state is not None:
            hidden_state.key_cache.data[:, dones] = torch.nan
            hidden_state.val_cache.data[:, dones] = torch.nan
        return self.backbone.reset_hidden_state(hidden_state, dones)

    def _history(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: TformerHiddenState | None,
        log_dict: dict[str, Any] | None,
    ) -> tuple[HistoryPacket, torch.Tensor, TformerHiddenState | None]:
        packet = HistoryPacket.unpack(
            seq, token_dim=self.token_dim, state_dim=self.state_dim
        )
        if hidden_state is not None and (seq.shape[1] != 1 or not packet.valid.all()):
            raise ContractError(
                "Online history cache accepts one valid decision per row."
            )
        history, hidden = self.backbone(
            packet.memory.contiguous(), time_idxs, hidden_state, log_dict
        )
        return packet, history, hidden

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: TformerHiddenState | None = None,
        log_dict: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, TformerHiddenState | None]:
        packet, history, hidden = self._history(seq, time_idxs, hidden_state, log_dict)
        control = (
            self.fusion(torch.cat((history, packet.state), -1))
            if self.bypass
            else history
        )
        return torch.where(packet.valid, control, 0.0), hidden


@gin.configurable
class DATTrajEncoder(TrajEncoder):  # type: ignore[misc]
    """Transition-history carrier whose selected blocks run dual attention.

    Construction deliberately builds a complete ordinary AMAGO backbone first,
    under the same initialization stream a baseline carrier would use, and only
    then replaces the selected blocks' attention. Two matched conditions
    therefore share byte-identical input, positional, unselected-block,
    feed-forward and final-normalization parameters at step zero; the selected
    attention modules are the intended difference.

    Only the no-bypass transition packet is supported, so the whole packet width
    is history: ``[token | valid]``.
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        spec: DATSpec | None = None,
        token_dim: int = TOKEN_DIM,
        d_model: int = 256,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__(tstep_dim, max_seq_len)
        if spec is None:
            raise ContractError("The DAT carrier requires a resolved attention spec.")
        if tstep_dim != token_dim + 1:
            raise ContractError(
                "DAT conditions run without the state bypass; the packet must be "
                "[token | valid]."
            )
        if spec.d_model != d_model:
            raise ContractError("DAT attention width disagrees with the carrier.")
        self.spec = spec
        self.token_dim = token_dim
        self.state_dim = 0
        self.output_dim = d_model
        self.capacity = max_seq_len + 1
        donor = TformerTrajEncoder(tstep_dim=token_dim, max_seq_len=self.capacity)
        if donor.emb_dim != d_model:
            raise ContractError("DAT backbone/output width mismatch.")
        with module_seed(initialization_seed, "dat-attention"):
            self.backbone = DATTransformer(donor.tformer, spec)
        self.register_buffer(
            "attention_protocol_identity",
            torch.tensor(list(bytes.fromhex(spec.sha256)), dtype=torch.uint8),
        )

    @property
    def emb_dim(self) -> int:
        return self.output_dim

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> DATHiddenState:
        return self.backbone.init_hidden_state(batch_size, device, self.capacity)

    def reset_hidden_state(
        self, hidden_state: DATHiddenState | None, dones: np.ndarray
    ) -> DATHiddenState | None:
        if hidden_state is not None:
            hidden_state.reset(dones)
        return hidden_state

    def _unpack(self, seq: torch.Tensor) -> HistoryPacket:
        return HistoryPacket.unpack(
            seq, token_dim=self.token_dim, state_dim=self.state_dim
        )

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: DATHiddenState | None = None,
        log_dict: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, DATHiddenState | None]:
        del log_dict
        packet = self._unpack(seq)
        if hidden_state is not None:
            if seq.shape[1] != 1 or not packet.valid.all():
                raise ContractError(
                    "Online DAT cache accepts one valid decision per row."
                )
            control, hidden = self.backbone(
                packet.memory.contiguous(), time_idxs, hidden_state
            )
            hidden_state.update()
            return torch.where(packet.valid, control, 0.0), hidden
        control, _ = self.backbone(
            packet.memory.contiguous(),
            time_idxs,
            None,
            valid=packet.valid.squeeze(-1),
        )
        return torch.where(packet.valid, control, 0.0), None

    @torch.no_grad()
    def rebuild_hidden_state(
        self,
        packets: torch.Tensor,
        time_idxs: torch.Tensor,
        lengths: Sequence[int],
    ) -> DATHiddenState:
        """Rebuild rollout caches from complete processed prefixes.

        Called after a learner update, when the weights and the timestep
        encoder's normalization statistics have both moved. Rows are replayed
        one at a time so that no padded decision is ever processed: feeding
        dummy tokens and afterwards resetting only the ordinary sequence lengths
        would leave relational and symbol state contaminated.
        """
        rows, _, _ = packets.shape
        if len(lengths) != rows:
            raise ContractError("DAT rebuild needs one true length per row.")
        state = self.init_hidden_state(rows, packets.device)
        for row, length in enumerate(lengths):
            if length <= 0:
                continue
            if length > self.capacity:
                raise ContractError("DAT rebuild prefix exceeds the cache capacity.")
            single = self.init_hidden_state(1, packets.device)
            for step in range(int(length)):
                self.backbone(
                    self._unpack(packets[row : row + 1, step : step + 1]).memory,
                    time_idxs[row : row + 1, step : step + 1],
                    single,
                )
                single.update()
            _copy_row(single, 0, state, row)
        return state


def _copy_row(
    source: DATHiddenState, source_row: int, target: DATHiddenState, target_row: int
) -> None:
    """Move one environment's rebuilt cache into the shared rollout state."""
    target.lengths[target_row] = source.lengths[source_row]
    target.times[target_row] = source.times[source_row]
    for src_cache, dst_cache in zip(source.layers, target.layers, strict=True):
        src_tensors = src_cache.tensors()
        for name, tensor in dst_cache.tensors().items():
            tensor[target_row] = src_tensors[name][source_row]


@gin.configurable
class GRUHistoryTrajEncoder(TrajEncoder):  # type: ignore[misc]
    """AMAGO's recurrent carrier over the packet's token columns.

    The grounding baseline of the summary-memory study: AMAGO's own
    ``GRUTrajEncoder`` at the study's width and depth, exactly what
    ``switch_traj_encoder("rnn", memory_size=width, layers=layers)`` would
    configure. It is not capacity-matched to the Transformer cells and runs no
    attention, no summary state and no dual-attention block. Only the no-bypass
    packet ``[token | valid]`` is supported.

    The rollout state is the GRU's own ``[n_layers, B, d_model]`` float32
    tensor. ``init_hidden_state`` returns zeros rather than AMAGO's ``None`` so
    that resets, the cache refresh and the serializer always see a tensor.
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        token_dim: int = TOKEN_DIM,
        d_model: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__(tstep_dim, max_seq_len)
        if tstep_dim != token_dim + 1:
            raise ContractError(
                "The GRU carrier runs without the state bypass; the packet must be "
                "[token | valid]."
            )
        if d_model <= 0 or n_layers <= 0:
            raise ContractError("GRU width and depth must be positive.")
        self.token_dim = token_dim
        self.state_dim = 0
        self.output_dim = d_model
        self.n_layers = n_layers
        self.backbone = GRUTrajEncoder(
            tstep_dim=token_dim,
            max_seq_len=max_seq_len + 1,
            d_hidden=d_model,
            n_layers=n_layers,
            d_output=d_model,
            norm="layer",
        )

    @property
    def emb_dim(self) -> int:
        return self.output_dim

    def init_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.output_dim, device=device)

    def reset_hidden_state(
        self, hidden_state: torch.Tensor | None, dones: Any
    ) -> torch.Tensor:
        """Zero the finished rows and leave every other row untouched."""
        if hidden_state is None:
            raise ContractError("The GRU carrier always carries a hidden state.")
        rows = torch.as_tensor(np.asarray(dones), device=hidden_state.device)
        if rows.dtype == torch.bool:
            rows = torch.where(rows.reshape(-1))[0]
        # The evaluator steps the policy under ``torch.inference_mode`` and resets
        # at task boundaries outside it; an inference tensor may only be written
        # in place inside that mode, so match the state's own mode.
        with torch.inference_mode(hidden_state.is_inference()):
            hidden_state[:, rows.long()] = 0.0
        return hidden_state

    def _unpack(self, seq: torch.Tensor) -> HistoryPacket:
        return HistoryPacket.unpack(
            seq, token_dim=self.token_dim, state_dim=self.state_dim
        )

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        log_dict: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the token columns; positions are implicit in the recurrence."""
        del log_dict
        packet = self._unpack(seq)
        if hidden_state is not None and (seq.shape[1] != 1 or not packet.valid.all()):
            raise ContractError("Online GRU state accepts one valid decision per row.")
        history, hidden = self.backbone(
            packet.memory.contiguous(), time_idxs, hidden_state
        )
        control = torch.where(packet.valid, history, 0.0)
        return control, (None if hidden_state is None else hidden)

    @torch.no_grad()
    def rebuild_hidden_state(
        self,
        packets: torch.Tensor,
        time_idxs: torch.Tensor,
        lengths: Sequence[int],
    ) -> torch.Tensor:
        """Recompute every actor's state from its true prefix, batched and exact.

        Called after a learner update. Rows are packed by their true lengths so
        no padded decision is ever processed; a row of length zero keeps the
        zero state.
        """
        del time_idxs
        rows, columns, _ = packets.shape
        if len(lengths) != rows:
            raise ContractError("GRU rebuild needs one true length per row.")
        state = self.init_hidden_state(rows, packets.device)
        active = [row for row, length in enumerate(lengths) if int(length) > 0]
        if not active:
            return state
        true_lengths = [int(lengths[row]) for row in active]
        if max(true_lengths) > columns:
            raise ContractError("GRU rebuild prefix exceeds the provided packets.")
        packed = pack_padded_sequence(
            self._unpack(packets[active]).memory.contiguous(),
            torch.as_tensor(true_lengths, dtype=torch.int64),
            batch_first=True,
            enforce_sorted=False,
        )
        _, final = self.backbone.rnn(packed)
        state[:, active] = final
        return state


@gin.configurable
class SummaryTrajEncoder(TrajEncoder):  # type: ignore[misc]
    """Segment or recurrent-summary carrier over the raw packet's token columns.

    Built exactly as the DAT carrier is: a complete ordinary AMAGO backbone
    first, under the same initialization stream every carrier uses, and then
    the segment/summary model adopts its modules. The ordinary blocks are the
    donor's own, so at step zero they are byte-identical to the full-prefix
    carrier's; the summary tables, write queries, memory projection and (for
    ``raw_dat_*`` cells) the selected dual-attention block are the difference.

    Only the no-bypass packet ``[token | valid]`` is supported. Positions are
    segment-local slot indices, so ``time_idxs`` are accepted and ignored.
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        spec: SummarySpec | None = None,
        dat: DATSpec | None = None,
        token_dim: int = TOKEN_DIM,
        d_model: int = 256,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__(tstep_dim, max_seq_len)
        if spec is None:
            raise ContractError("The summary carrier requires a resolved summary spec.")
        if tstep_dim != token_dim + 1:
            raise ContractError(
                "Summary cells run without the state bypass; the packet must be "
                "[token | valid]."
            )
        if spec.d_model != d_model:
            raise ContractError("Summary width disagrees with the carrier.")
        if dat is not None and dat.d_model != d_model:
            raise ContractError("DAT attention width disagrees with the carrier.")
        self.spec = spec
        self.dat = dat
        self.token_dim = token_dim
        self.state_dim = 0
        self.output_dim = d_model
        self.capacity = spec.capacity
        donor = TformerTrajEncoder(tstep_dim=token_dim, max_seq_len=max_seq_len + 1)
        if donor.emb_dim != d_model:
            raise ContractError("Summary backbone/output width mismatch.")
        with module_seed(initialization_seed, "summary-memory"):
            self.backbone = SummaryTransformer(donor.tformer, spec, dat)
        self.register_buffer(
            "summary_protocol_identity",
            torch.tensor(list(bytes.fromhex(spec.sha256)), dtype=torch.uint8),
        )
        if dat is not None:
            self.register_buffer(
                "attention_protocol_identity",
                torch.tensor(list(bytes.fromhex(dat.sha256)), dtype=torch.uint8),
            )

    @property
    def emb_dim(self) -> int:
        return self.output_dim

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> SummaryHiddenState:
        return self.backbone.init_hidden_state(batch_size, device)

    def reset_hidden_state(
        self, hidden_state: SummaryHiddenState | None, dones: Any
    ) -> SummaryHiddenState | None:
        if hidden_state is not None:
            hidden_state.reset(np.asarray(dones))
        return hidden_state

    def _unpack(self, seq: torch.Tensor) -> HistoryPacket:
        return HistoryPacket.unpack(
            seq, token_dim=self.token_dim, state_dim=self.state_dim
        )

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: SummaryHiddenState | None = None,
        log_dict: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, SummaryHiddenState | None]:
        """Whole tasks without a state; one valid record per row with one."""
        del time_idxs, log_dict
        packet = self._unpack(seq)
        if hidden_state is not None:
            if seq.shape[1] != 1 or not packet.valid.all():
                raise ContractError(
                    "Online summary state accepts one valid decision per row."
                )
            control = self.backbone.cached_forward(
                packet.memory.contiguous(), hidden_state
            )
            return torch.where(packet.valid, control, 0.0), hidden_state
        control, _ = self.backbone.training_forward(
            packet.memory.contiguous(), packet.valid.squeeze(-1)
        )
        return torch.where(packet.valid, control, 0.0), None

    @torch.no_grad()
    def rebuild_hidden_state(
        self,
        packets: torch.Tensor,
        time_idxs: torch.Tensor,
        lengths: Sequence[int],
    ) -> SummaryHiddenState:
        """Recompute every actor's state from its true prefix under current weights.

        Called after a learner update. Full segments run through the dense
        path, the open segment through the cached path; no padded record is
        ever processed and the initial memory is refreshed from the weights.
        """
        del time_idxs
        return self.backbone.rebuild(self._unpack(packets).memory.contiguous(), lengths)


@gin.configurable
class WindowTrajEncoder(TrajEncoder):  # type: ignore[misc]
    """Sliding-window carrier over the raw packet's token columns (SPEC §5).

    The AMAGO backbone, built exactly as the full-prefix carrier builds it so
    that every unselected block is byte-identical to ``full_context``'s at step
    zero, run under a per-layer band of ``W`` slots: a banded mask in training
    and a ``W``-slot rolling cache per layer at rollout. Without ``dat`` this
    is the legacy ordinary window (``amago-window-v1``) and adds no parameter;
    with ``dat`` the selected blocks run dual attention over the band
    (``amago-dat-window-v1``, the 8M study's ``fixed_window``), built under an
    isolated initialization stream and with the symbol table clipped at ``W``.
    Only the no-bypass packet ``[token | valid]`` is supported; positions are
    the true trajectory times.
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        spec: WindowSpec | None = None,
        dat: DATSpec | None = None,
        token_dim: int = TOKEN_DIM,
        d_model: int = 256,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__(tstep_dim, max_seq_len)
        if spec is None:
            raise ContractError("The window carrier requires a resolved window spec.")
        if tstep_dim != token_dim + 1:
            raise ContractError(
                "The window cell runs without the state bypass; the packet must be "
                "[token | valid]."
            )
        if dat is not None and dat.d_model != d_model:
            raise ContractError("DAT attention width disagrees with the carrier.")
        self.spec = spec
        self.dat = dat
        self.token_dim = token_dim
        self.state_dim = 0
        self.output_dim = d_model
        self.capacity = spec.capacity
        donor = TformerTrajEncoder(tstep_dim=token_dim, max_seq_len=max_seq_len + 1)
        if donor.emb_dim != d_model:
            raise ContractError("Window backbone/output width mismatch.")
        if dat is None:
            self.backbone = WindowTransformer(donor.tformer, spec)
        else:
            with module_seed(initialization_seed, "dat-window"):
                self.backbone = WindowTransformer(donor.tformer, spec, dat)
        self.register_buffer(
            "window_protocol_identity",
            torch.tensor(list(bytes.fromhex(spec.sha256)), dtype=torch.uint8),
        )
        if dat is not None:
            self.register_buffer(
                "attention_protocol_identity",
                torch.tensor(list(bytes.fromhex(dat.sha256)), dtype=torch.uint8),
            )

    @property
    def emb_dim(self) -> int:
        return self.output_dim

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> DATHiddenState:
        return self.backbone.init_hidden_state(batch_size, device)

    def reset_hidden_state(
        self, hidden_state: DATHiddenState | None, dones: Any
    ) -> DATHiddenState | None:
        if hidden_state is not None:
            hidden_state.reset(np.asarray(dones))
        return hidden_state

    def _unpack(self, seq: torch.Tensor) -> HistoryPacket:
        return HistoryPacket.unpack(
            seq, token_dim=self.token_dim, state_dim=self.state_dim
        )

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: DATHiddenState | None = None,
        log_dict: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, DATHiddenState | None]:
        """Whole tasks without a state; one valid record per row with one."""
        del log_dict
        packet = self._unpack(seq)
        times = time_idxs.squeeze(-1)
        if hidden_state is not None:
            if seq.shape[1] != 1 or not packet.valid.all():
                raise ContractError(
                    "Online window state accepts one valid decision per row."
                )
            control = self.backbone.cached_forward(
                packet.memory.contiguous(), times, hidden_state
            )
            hidden_state.update()
            return torch.where(packet.valid, control, 0.0), hidden_state
        control = self.backbone.training_forward(
            packet.memory.contiguous(), times, packet.valid.squeeze(-1)
        )
        return torch.where(packet.valid, control, 0.0), None

    @torch.no_grad()
    def rebuild_hidden_state(
        self,
        packets: torch.Tensor,
        time_idxs: torch.Tensor,
        lengths: Sequence[int],
    ) -> DATHiddenState:
        """Recompute every actor's cache from its true prefix under current weights."""
        return self.backbone.rebuild(
            self._unpack(packets).memory.contiguous(), time_idxs.squeeze(-1), lengths
        )


@gin.configurable
class MemoTrajEncoder(TrajEncoder):  # type: ignore[misc]
    """The Memo comparator over the raw packet's token columns (ME0/ME1).

    Built exactly as the summary carrier is: a complete ordinary AMAGO backbone
    first, under the same initialization stream every carrier uses, and then
    the Memo model adopts its modules, so every block, the input projection,
    the position table and the final norm are byte-identical to
    ``full_context``'s at step zero; the ``S`` summary embeddings are the only
    difference. The cache is allocated for ``max_seq_len + 1`` records, the
    longest task of the contract, and its live length grows with the
    boundaries crossed. Only the no-bypass packet ``[token | valid]`` is
    supported. Positions are block slot indices, so ``time_idxs`` are accepted
    and ignored.
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        spec: MemoSpec | None = None,
        token_dim: int = TOKEN_DIM,
        d_model: int = 256,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__(tstep_dim, max_seq_len)
        if spec is None:
            raise ContractError("The Memo carrier requires a resolved Memo spec.")
        if tstep_dim != token_dim + 1:
            raise ContractError(
                "The Memo cell runs without the state bypass; the packet must be "
                "[token | valid]."
            )
        if spec.d_model != d_model:
            raise ContractError("Memo width disagrees with the carrier.")
        self.spec = spec
        self.dat = None
        self.token_dim = token_dim
        self.state_dim = 0
        self.output_dim = d_model
        self.capacity = spec.capacity(max_seq_len)
        donor = TformerTrajEncoder(tstep_dim=token_dim, max_seq_len=max_seq_len + 1)
        if donor.emb_dim != d_model:
            raise ContractError("Memo backbone/output width mismatch.")
        with module_seed(initialization_seed, "memo-summary"):
            self.backbone = MemoTransformer(donor.tformer, spec, max_index=max_seq_len)
        self.register_buffer(
            "memo_protocol_identity",
            torch.tensor(list(bytes.fromhex(spec.sha256)), dtype=torch.uint8),
        )

    @property
    def emb_dim(self) -> int:
        return self.output_dim

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> MemoHiddenState:
        return self.backbone.init_hidden_state(batch_size, device)

    def reset_hidden_state(
        self, hidden_state: MemoHiddenState | None, dones: Any
    ) -> MemoHiddenState | None:
        if hidden_state is not None:
            hidden_state.reset(np.asarray(dones))
        return hidden_state

    def _unpack(self, seq: torch.Tensor) -> HistoryPacket:
        return HistoryPacket.unpack(
            seq, token_dim=self.token_dim, state_dim=self.state_dim
        )

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: MemoHiddenState | None = None,
        log_dict: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, MemoHiddenState | None]:
        """Whole tasks without a state; one valid record per row with one."""
        del time_idxs, log_dict
        packet = self._unpack(seq)
        if hidden_state is not None:
            if seq.shape[1] != 1 or not packet.valid.all():
                raise ContractError(
                    "Online Memo state accepts one valid decision per row."
                )
            control = self.backbone.cached_forward(
                packet.memory.contiguous(), hidden_state
            )
            return torch.where(packet.valid, control, 0.0), hidden_state
        control, _ = self.backbone.training_forward(
            packet.memory.contiguous(), packet.valid.squeeze(-1)
        )
        return torch.where(packet.valid, control, 0.0), None

    @torch.no_grad()
    def rebuild_hidden_state(
        self,
        packets: torch.Tensor,
        time_idxs: torch.Tensor,
        lengths: Sequence[int],
    ) -> MemoHiddenState:
        """Recompute every actor's state from its true prefix under current weights.

        Called after a learner update: the completed segments run through the
        dense path at the fixed ``L`` and their summaries are entered into the
        caches, the open segment through the cached path; no padded record is
        ever processed.
        """
        del time_idxs
        return self.backbone.rebuild(self._unpack(packets).memory.contiguous(), lengths)


__all__ = [
    "TOKEN_DIM",
    "DATTrajEncoder",
    "GRUHistoryTrajEncoder",
    "HistoryPacket",
    "HistoryTrajEncoder",
    "MemoTrajEncoder",
    "SummaryTrajEncoder",
    "WindowTrajEncoder",
]
