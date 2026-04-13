"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

from torch import nn
from torch.nn import functional as F


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 256,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size)

        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, d_model))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = d_model
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        pred = self.net(state)
        return pred.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred = self.forward(state)
        return F.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 256,
        depth: int = 3,
        dropout: float = 0.0,
        state_keys: list[str] | None = None,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size)

        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, d_model))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = d_model
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred = self.forward(state)
        return F.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        pred = self.net(state)
        return pred.view(-1, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 16,
    d_model: int = 256,
    depth: int = 3,
    dropout: float = 0.0,
    state_keys: list[str] | None = None,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
            state_keys=state_keys,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
