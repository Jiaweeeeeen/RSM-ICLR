"""Project-owned AMAGO learner with component-isolated initialization.

Every condition shares the native AMAGO v3.4 objectives and heads; only the
timestep and trajectory encoders differ. Seeding each component separately
keeps matched comparisons matched: two conditions with the same seed draw the
same head weights regardless of how many parameters their encoders use.
"""

from __future__ import annotations

from typing import Any, cast

import gin
import torch
from amago.agent import Agent
from amago.nets.tstep_encoders import FFTstepEncoder

from reasoned_icrl.model.utils import module_seed, reset_named_linears

__all__ = ["MatchedBaselineAgent"]


@gin.configurable
class MatchedBaselineAgent(Agent):  # type: ignore[misc]
    """Native AMAGO objectives and heads with component-isolated initialization."""

    def __init__(
        self,
        *args: Any,
        initialization_seed: int = 0,
        query_loss_field: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.initialization_seed = initialization_seed
        self.query_loss_field = query_loss_field
        with module_seed(initialization_seed, "agent"):
            super().__init__(*args, **kwargs)

    def _query_mask(self, batch: Any, mask: torch.Tensor) -> torch.Tensor:
        if self.query_loss_field is None:
            return mask
        query = batch.obs["current"][:, :-1, self.query_loss_field] == 1
        query = query & (batch.obs["valid"][:, :-1, 0] == 1)
        return cast(
            torch.Tensor, mask & query.reshape(*query.shape, *((1,) * (mask.ndim - 2)))
        )

    def edit_actor_mask(
        self, batch: Any, actor_loss: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        return self._query_mask(batch, pad_mask)

    def edit_critic_mask(
        self, batch: Any, critic_loss: torch.Tensor | None, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        return self._query_mask(batch, pad_mask)

    def init_encoders(self) -> None:
        with module_seed(self.initialization_seed, "timestep-construction"):
            self.tstep_encoder = self.tstep_encoder_type(self.obs_space, self.rl2_space)
            if isinstance(self.tstep_encoder, FFTstepEncoder):
                reset_named_linears(
                    self.tstep_encoder, self.initialization_seed, "timestep"
                )
        with module_seed(self.initialization_seed, "trajectory"):
            self.traj_encoder = self.traj_encoder_type(
                tstep_dim=self.tstep_encoder.emb_dim,
                max_seq_len=self.max_seq_len,
            )

    def init_actor_critic(self) -> None:
        with module_seed(self.initialization_seed, "heads"):
            super().init_actor_critic()
