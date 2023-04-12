from typing import Union

import gym

import torch
import torch.nn as nn

from core.networks.modules.distributions import Categorical, DiagonalGaussian
from core.networks.base_actor import BaseActor


class StatelessActor(BaseActor):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_size: int = 64,
    ):
        """
        Actor-Critic for a discrete action space.

        Args:
          observation_space (gym.Space): State dimensions for the environment.
          action_space (gym.Space): Action space in which the agent is operating.
        """
        super(StatelessActor, self).__init__(observation_space, action_space)

        self._mlp = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        if action_space.__class__.__name__ == "Discrete":
            self._dist = Categorical(hidden_size, action_space.n)
        elif action_space.__class__.__name__ == "Box":
            self._dist = DiagonalGaussian(hidden_size, action_space.shape[0])

    def forward(self, x: torch.Tensor) -> Union[Categorical, DiagonalGaussian]:
        """
        Conduct the forward pass through the network.

        Args:
          x (torch.Tensor): Input for the forward pass.

        Returns:
          Union[Categorical, DiagonalGaussian]
        """
        x = self._mlp(x)

        return self._dist(x)
