from typing import Tuple

import torch
import torch.nn as nn

from core.networks.base_actor_critic import BaseActorCritic
from core.learners.rollout_storage import RolloutStorage


class PPO:

    OPT_ACTOR_PARAMS = "params:actor"
    OPT_CRITIC_PARAMS = "params:critic"

    def __init__(
        self,
        actor_critic: BaseActorCritic,
        clip_param: float,
        num_epochs: int,
        num_minibatches: int,
        entropy_coef: float,
        value_loss_coef: float,
        actor_lr: float,
        critic_lr: float,
        eps: float = None,
        max_grad_norm: float = None,
        use_clipped_value_loss: bool = True,
    ):
        """
        PPO implementation based on "Proximal Policy Optimization Algorithms" https://arxiv.org/abs/1707.06347.

        Args:
            actor_critic (BaseActorCritic): Actor-Critic to train with PPO.
            clip_param (float): Clip param for PPO.
            num_epochs (int): Number of epochs to train over.
            num_minibatches (int): Number of minibatches for training.
            entropy_coef (float): Entropy coefficient to be used while computing the loss.
            value_loss_coef (float): Value loss coefficient to be used while computing the loss.
            actor_lr (float): Learning rate of the actor network.
            critic_lr (float): Learning rate of the critic network.
            eps (float): Epsilon value to use with the Adam optimizer.
            max_grad_norm (float): Max gradient norm for gradient clipping.
            use_clipped_value_loss (bool): Whether to use the clipped value loss while computing the objective.
        """
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.initial_actor_lr = actor_lr

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.actor_critic.actor.parameters(),
                    "lr": actor_lr,
                    "eps": eps,
                    "name": self.OPT_ACTOR_PARAMS,
                },
                {
                    "params": self.actor_critic.critic.parameters(),
                    "lr": critic_lr,
                    "eps": eps,
                    "name": self.OPT_CRITIC_PARAMS,
                },
            ]
        )
        pass

    def update_linear_schedule(self, current_epoch: int, total_epochs: int):
        """
        Update linear schedule for the actor's learning rate.

        Args:
            current_epoch (int): Current training epoch.
            total_epochs (int): Total epochs over which to decay the learning rate.

        Returns:
            None
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] != self.OPT_ACTOR_PARAMS:
                continue

            lr = self.initial_actor_lr - (
                self.initial_actor_lr * (current_epoch / float(total_epochs))
            )
            param_group["lr"] = lr
            pass

    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        """
        Update the policy and value function.

        Args:
          rollouts (RolloutStorage): Rollouts to be used as data points for making updates.

        Returns:
          Tuple[float, float, float]
        """
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.num_epochs):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_minibatches
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_minibatches
                )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.actor.parameters(), self.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.actor_critic.critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.num_epochs * self.num_minibatches

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
