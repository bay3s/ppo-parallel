from typing import Tuple
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import gym


def _flatten_helper(T: int, N: int, _tensor: torch.Tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage:

    def __init__(
        self,
        rollout_steps: int,
        num_processes: int,
        obs_shape: Tuple,
        action_space: gym.Space,
        recurrent_state_size: int,
    ):
        """
        Initialize the rollout storage.

        Args:
            rollout_steps (int): Number of *steps per rollout*.
            num_processes (int): Number of parallel processes.
            obs_shape (Tuple): Observation shape.
            action_space (gym.Space): Action space for the environment.
            recurrent_state_size (int): Recurrent state size for a memory-augmented agent.
        """
        self.obs = torch.zeros(rollout_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            rollout_steps + 1, num_processes, recurrent_state_size
        )
        self.rewards = torch.zeros(rollout_steps, num_processes, 1)
        self.value_preds = torch.zeros(rollout_steps + 1, num_processes, 1)
        self.returns = torch.zeros(rollout_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(rollout_steps, num_processes, 1)

        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
            self.actions = torch.zeros(rollout_steps, num_processes, action_shape).long()
        elif action_space.__class__.__name__ == "Box":
            action_shape = action_space.shape[0]
            self.actions = torch.zeros(rollout_steps, num_processes, action_shape)
        else:
            raise NotImplementedError

        self.done_masks = torch.ones(rollout_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state or time limit end state
        self.time_limit_masks = torch.ones(rollout_steps + 1, num_processes, 1)

        self.rollout_steps = rollout_steps
        self.step = 0

    def to(self, device: torch.device) -> None:
        """
        Transfer the tensors to a specific device.

        Args:
            device (torch.device): Torch device on which to transfer the tensors.

        Returns:
            None
        """
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.done_masks = self.done_masks.to(device)
        self.time_limit_masks = self.time_limit_masks.to(device)
        pass

    def insert(
        self,
        obs: torch.Tensor,
        recurrent_hidden_states: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        done_masks: torch.Tensor,
        time_limit_masks: torch.Tensor,
    ):
        """
        Insert a transtions details into the rollout storage.

        Args:
            obs (torch.Tensor): Observations to be inserted.
            recurrent_hidden_states (torch.Tensor): Recurrent hidden states to be inserted.
            actions (torch.Tensor): Actions to be inserted.
            action_log_probs (torch.Tensor): Log probabilities of actions to be inserted.
            value_preds (torch.Tensor): Value predictions to be inserted.
            rewards (torch.Tensor): Rewards to be inserted.
            done_masks (torch.Tensor): Done masks to be inserted (0 if done, 1 if not).
            time_limit_masks (torch.Tensor): Time limit masks to be inserted (0 if time-limit is hit, 1 if not).

        Returns:
            None
        """
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.done_masks[self.step + 1].copy_(done_masks)
        self.time_limit_masks[self.step + 1].copy_(time_limit_masks)

        self.step = (self.step + 1) % self.rollout_steps

    def after_update(self):
        """
        Post-update processing to be done once PPO `update` is called.

        Returns:
            None
        """
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.done_masks[0].copy_(self.done_masks[-1])
        self.time_limit_masks[0].copy_(self.time_limit_masks[-1])

    def compute_returns(
        self,
        next_value: torch.Tensor,
        use_gae: bool,
        gamma: float,
        gae_lambda: float,
        use_proper_time_limits: bool = True
    ) -> None:
        """
        Compute returns for each of the rollouts.

        Args:
            next_value (torch.Tensor): Next predicted value.
            use_gae (bool): Whether to use GAE for advantage estimates.
            gamma (float): Discount gamme to be used.
            gae_lambda (float): GAE lambda value.
            use_proper_time_limits (bool): Whether to use proper time limits for end of episode.

        Returns:
            None
        """
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.done_masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + gamma * gae_lambda * self.done_masks[step + 1] * gae
                    gae = gae * self.time_limit_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.done_masks[step + 1]
                        + self.rewards[step]
                    ) * self.time_limit_masks[step + 1] + (
                        1 - self.time_limit_masks[step + 1]
                    ) * self.value_preds[
                        step
                    ]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.done_masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + gamma * gae_lambda * self.done_masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.done_masks[step + 1]
                        + self.rewards[step]
                    )

    def feed_forward_generator(
        self,
        advantages: torch.Tensor,
        num_minibatches: int = None,
        minibatch_size: int = None
    ):
        """
        Mini-batch generator for training the policy of an agent.

        Args:
            advantages (torch.Tensor): Computed advatages.
            num_minibatches (int): Number of minibatches.
            minibatch_size (int): Mini-batch size on which to train the agent.

        Yields:
            Tuple
        """
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if minibatch_size is None:
            assert batch_size >= num_minibatches, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    num_processes, num_steps, num_processes * num_steps, num_minibatches
                )
            )
            minibatch_size = batch_size // num_minibatches
            pass

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=True
        )

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1)
            )[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.done_masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages: torch.Tensor, num_minibatches: int):
        """
        Mini-batch generator for training the policy of an agent.

        Args:
            advantages (torch.Tensor): Computed advatages.
            num_minibatches (int): Number of minibatches.

        Yields:
            Tuple
        """
        num_processes = self.rewards.size(1)
        assert num_processes >= num_minibatches, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_minibatches)
        )

        num_envs_per_batch = num_processes // num_minibatches
        perm = torch.randperm(num_processes)

        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind]
                )
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.done_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                pass

            T, N = self.rollout_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            ).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ
