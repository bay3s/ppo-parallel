from typing import Tuple

import torch
import gym

from core.networks.stateless.stateless_actor import StatelessActor
from core.networks.stateless.stateless_critic import StatelessCritic

from core.networks.base_actor_critic import BaseActorCritic
from core.networks.base_critic import BaseCritic
from core.networks.base_actor import BaseActor


class StatelessActorCritic(BaseActorCritic):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """
        Actor-Critic for a discrete action space.

        Args:
          observation_space (gym.Space): Observation space in which the agent operates.
          action_space (gym.Space): Action space in which the agent operates.
        """
        super(StatelessActorCritic, self).__init__(observation_space, action_space)
        self._actor = StatelessActor(observation_space, action_space)
        self._critic = StatelessCritic(observation_space)
        pass

    @property
    def actor(self) -> BaseActor:
        """
        Return the actor network.

        Returns:
          BaseActor
        """
        return self._actor

    @property
    def critic(self) -> BaseCritic:
        """
        Return the critic network.

        Returns:
          BaseCritic
        """
        return self._critic

    def to_device(self, device: torch.device) -> "StatelessActorCritic":
        """
        Performs device conversion on the actor and critic.

        Returns:
          StatelessActorCritic
        """
        self._actor.to(device)
        self._critic.to(device)

        return self

    def act(
        self,
        observations: torch.Tensor,
        recurrent_states: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a state return the action to take, log probability of said action, and the current state value
        computed by the critic.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states (torch.Tensor): Recurrent states that are being used in memory-based policies.
          masks (torch.Tensor): Masks based on terminal states.
          deterministic (bool): Whether to choose actions deterministically.

        Returns:
          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        value = self.critic(observations)

        distribution = self.actor(observations)
        actions = distribution.mode() if deterministic else distribution.sample()

        return value, actions, distribution.log_probs(actions), recurrent_states

    def get_value(
        self,
        observations: torch.Tensor,
        recurrent_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given a state returns its corresponding value.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states (torch.Tensor): Recurrent states that are being used in memory-based policies.
          masks (torch.Tensor): Masks based on terminal states.

        Returns:
          torch.Tensor
        """
        return self.critic(observations)

    def evaluate_actions(self, inputs, recurrent_states, masks, actions) -> Tuple:
        """
        Evaluate actions given observations, encoded states, done_masks, actions.

        Returns:
          Tuple
        """
        value = self.critic(inputs)

        dist = self.actor(inputs)
        log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return value, log_probs, dist_entropy, recurrent_states

    @property
    def recurrent_state_size(self) -> int:
        """
        Returns the size of the encoded state (eg. hidden state in a recurrent agent).

        Returns:
          int
        """
        return 1

    @property
    def is_recurrent(self) -> bool:
        """
        Whether the actor critic is stateful / recurrent.

        Returns:
          bool
        """
        return False

    def forward(self) -> None:
        """
        Forward pass for the network, in this case not implemented.

        Returns:
          None
        """
        raise NotImplementedError
