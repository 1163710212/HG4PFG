import math

import torch

from model.buffer.BaseBuffer import BaseBuffer


class HRL4PFGBuffer(BaseBuffer):
    """On-policy storage at HRL4PFG's two temporal resolutions."""

    @staticmethod
    def parse_model_args(parser):
        return BaseBuffer.parse_model_args(parser)

    def __init__(self, args, environment, policy, critic):
        super().__init__(args, environment, policy, critic)
        self.subgoal_interval = args.subgoal_interval
        expected_low_size = args.train_every_n_step * environment.episode_batch_size
        if self.buffer_size < expected_low_size:
            raise ValueError(
                'buffer_size must be at least episode_batch_size * '
                'train_every_n_step to preserve the on-policy rollout'
            )

        # At most one macro transition can end per active user and environment
        # step.  The tighter M-based estimate is padded for early user exits.
        self.high_capacity = min(
            self.buffer_size,
            max(
                expected_low_size,
                int(math.ceil(float(self.buffer_size) / self.subgoal_interval))
                + 2 * environment.episode_batch_size,
            ),
        )

    @staticmethod
    def _copy_observation(target, target_indices, source, source_indices=None):
        for group in ('user_profile', 'user_history'):
            for key in target[group]:
                values = source[group][key]
                if source_indices is not None:
                    values = values[source_indices]
                target[group][key][target_indices] = values.detach()

    def reset(self, environment, actor):
        self.low_observation = environment.create_observation_buffer(self.buffer_size)
        self.low_next_observation = environment.create_observation_buffer(
            self.buffer_size
        )
        self.high_observation = environment.create_observation_buffer(
            self.high_capacity
        )
        self.high_next_observation = environment.create_observation_buffer(
            self.high_capacity
        )

        device = self.device
        self.low_goal = torch.zeros(self.buffer_size, actor.goal_dim, device=device)
        self.low_next_goal = torch.zeros(
            self.buffer_size, actor.goal_dim, device=device
        )
        self.low_state = torch.zeros(
            self.buffer_size, 2 * actor.goal_dim, device=device
        )
        self.low_action = torch.zeros(
            self.buffer_size,
            actor.effect_action_dim,
            dtype=torch.long,
            device=device,
        )
        self.low_old_log_prob = torch.zeros(self.buffer_size, device=device)
        self.low_reward = torch.zeros(self.buffer_size, device=device)
        self.low_done = torch.zeros(
            self.buffer_size, dtype=torch.bool, device=device
        )

        self.high_prototype = torch.zeros(
            self.high_capacity, actor.goal_dim, device=device
        )
        self.high_old_log_prob = torch.zeros(self.high_capacity, device=device)
        self.high_reward = torch.zeros(self.high_capacity, device=device)
        self.high_done = torch.zeros(
            self.high_capacity, dtype=torch.bool, device=device
        )
        self.high_duration = torch.zeros(
            self.high_capacity, dtype=torch.long, device=device
        )
        self.clear()
        return self

    @staticmethod
    def _ring_indices(head, count, capacity, device):
        return (
            torch.arange(count, device=device, dtype=torch.long) + head
        ) % capacity

    def append_low(
        self,
        observation,
        next_observation,
        goal,
        next_goal,
        low_state,
        action,
        old_log_prob,
        reward,
        done,
    ):
        batch_size = int(reward.shape[0])
        if batch_size > self.buffer_size:
            raise ValueError('one low-level batch exceeds HRL4PFGBuffer capacity')
        indices = self._ring_indices(
            self.low_head, batch_size, self.buffer_size, self.device
        )
        self._copy_observation(self.low_observation, indices, observation)
        self._copy_observation(
            self.low_next_observation, indices, next_observation
        )
        self.low_goal[indices] = goal.detach()
        self.low_next_goal[indices] = next_goal.detach()
        self.low_state[indices] = low_state.detach()
        self.low_action[indices] = action.detach()
        self.low_old_log_prob[indices] = old_log_prob.detach().reshape(-1)
        self.low_reward[indices] = reward.detach().reshape(-1)
        self.low_done[indices] = done.detach().bool().reshape(-1)
        self.low_head = (self.low_head + batch_size) % self.buffer_size
        self.low_size = min(self.low_size + batch_size, self.buffer_size)

    def append_high(
        self,
        observation,
        next_observation,
        prototype,
        old_log_prob,
        reward,
        done,
        duration,
        source_indices,
    ):
        count = int(source_indices.numel())
        if count == 0:
            return
        if count > self.high_capacity:
            raise ValueError('one high-level batch exceeds HRL4PFGBuffer capacity')
        indices = self._ring_indices(
            self.high_head, count, self.high_capacity, self.device
        )
        self._copy_observation(
            self.high_observation, indices, observation, source_indices
        )
        self._copy_observation(
            self.high_next_observation, indices, next_observation, source_indices
        )
        self.high_prototype[indices] = prototype[source_indices].detach()
        self.high_old_log_prob[indices] = (
            old_log_prob[source_indices].detach().reshape(-1)
        )
        self.high_reward[indices] = reward[source_indices].detach().reshape(-1)
        self.high_done[indices] = done[source_indices].detach().bool().reshape(-1)
        self.high_duration[indices] = (
            duration[source_indices].detach().long().reshape(-1)
        )
        self.high_head = (self.high_head + count) % self.high_capacity
        self.high_size = min(self.high_size + count, self.high_capacity)

    @staticmethod
    def _slice_observation(observation, indices):
        return {
            group: {key: value[indices] for key, value in observation[group].items()}
            for group in ('user_profile', 'user_history')
        }

    def low_data(self):
        indices = torch.arange(
            self.low_size, device=self.device, dtype=torch.long
        )
        return {
            'observation': self._slice_observation(self.low_observation, indices),
            'next_observation': self._slice_observation(
                self.low_next_observation, indices
            ),
            'goal': self.low_goal[indices],
            'next_goal': self.low_next_goal[indices],
            'low_state': self.low_state[indices],
            'action': self.low_action[indices],
            'old_log_prob': self.low_old_log_prob[indices],
            'reward': self.low_reward[indices],
            'done': self.low_done[indices],
        }

    def high_data(self):
        indices = torch.arange(
            self.high_size, device=self.device, dtype=torch.long
        )
        return {
            'observation': self._slice_observation(self.high_observation, indices),
            'next_observation': self._slice_observation(
                self.high_next_observation, indices
            ),
            'prototype': self.high_prototype[indices],
            'old_log_prob': self.high_old_log_prob[indices],
            'reward': self.high_reward[indices],
            'done': self.high_done[indices],
            'duration': self.high_duration[indices],
        }

    def clear(self):
        self.low_head = 0
        self.low_size = 0
        self.high_head = 0
        self.high_size = 0
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.n_stream_record = 0
