"""Context-free UCB baseline for whole-session recommendation.

The implementation follows the context-free comparator described in
Li et al., *A Contextual-Bandit Approach to Personalized News Article
Recommendation*.  Arm ``a`` is scored as::

    empirical_mean[a] + alpha / sqrt(selection_count[a])

Unseen arms receive an infinite score.  For the simulator's list action, the
policy uses the standard semi-bandit extension: recommend the K arms with the
largest scores and update each selected arm from its observed item-level
feedback.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import model.reward as reward_functions
from model.agent.LinUCB import LinUCB


class UCB(LinUCB):
    """Context-free upper-confidence-bound recommender."""

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--reward_func', type=str, default='get_immediate_reward')
        parser.add_argument('--n_iter', type=int, nargs='+', default=[30000])
        parser.add_argument('--check_episode', type=int, default=10)
        parser.add_argument('--save_episode', type=int, default=200)
        parser.add_argument('--save_path', type=str, required=True)
        parser.add_argument('--test_n_step', type=int, default=100)
        parser.add_argument('--test_seed', type=int, default=2027)
        parser.add_argument('--test_repeat', type=int, default=1)
        parser.add_argument('--test_num_users', type=int, default=0)
        parser.add_argument(
            '--ucb_alpha', type=float, default=1.0,
            help='confidence coefficient in alpha / sqrt(n_a)',
        )
        return parser

    def __init__(self, args, env):
        if not math.isfinite(args.ucb_alpha) or args.ucb_alpha < 0.0:
            raise ValueError('ucb_alpha must be finite and non-negative')
        if args.check_episode <= 0 or args.save_episode <= 0:
            raise ValueError('check_episode and save_episode must be positive')
        if not args.n_iter or any(iterations <= 0 for iterations in args.n_iter):
            raise ValueError('every n_iter stage must be positive')

        self.device = torch.device(args.device)
        self.env = env
        # BaseRLAgent's shared final-test implementation expects an actor-like
        # module only for train/eval mode switching; UCB itself is tabular.
        self.actor = nn.Identity().to(self.device)
        try:
            self.reward_func = getattr(reward_functions, args.reward_func)
        except AttributeError as error:
            raise ValueError(f'Unknown reward function: {args.reward_func}') from error
        self.n_iter = list(args.n_iter)
        self.check_episode = args.check_episode
        self.save_episode = args.save_episode
        self.save_path = args.save_path
        self.test_n_step = args.test_n_step
        self.test_seed = args.test_seed
        self.test_repeat = args.test_repeat
        self.test_num_users = args.test_num_users
        self.alpha = float(args.ucb_alpha)
        self.n_arms = int(env.n_candidate)
        self.slate_size = int(env.slate_size)
        if self.slate_size > self.n_arms:
            raise ValueError('slate_size cannot exceed the number of arms')

        self.selection_counts = torch.zeros(
            self.n_arms, dtype=torch.long, device=self.device
        )
        self.reward_sums = torch.zeros(
            self.n_arms, dtype=torch.float32, device=self.device
        )
        self.total_updates = 0

        os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
        if len(self.n_iter) == 1:
            with open(self.save_path + '.report', 'w', encoding='utf-8') as report:
                report.write(str(args) + '\n')

    def score(self):
        """Return the context-free UCB score of every arm."""
        scores = torch.full(
            (self.n_arms,), float('inf'),
            dtype=torch.float32, device=self.device,
        )
        observed = self.selection_counts > 0
        counts = self.selection_counts[observed].to(torch.float32)
        means = self.reward_sums[observed] / counts
        scores[observed] = means + self.alpha / torch.sqrt(counts)
        return scores

    def select(self, batch_size):
        """Choose one context-free top-K slate for the whole user batch."""
        scores, indices = torch.topk(
            self.score(), k=self.slate_size, sorted=True
        )
        return (
            indices.view(1, -1).expand(batch_size, -1).clone(),
            scores.view(1, -1).expand(batch_size, -1).clone(),
        )

    def update(self, action_indices, arm_rewards):
        """Update counts and empirical reward sums from semi-bandit feedback."""
        if action_indices.shape != arm_rewards.shape:
            raise ValueError('action_indices and arm_rewards must have equal shape')
        arms = action_indices.reshape(-1)
        rewards = arm_rewards.reshape(-1).to(
            device=self.device, dtype=torch.float32
        )
        increments = torch.ones_like(arms, dtype=torch.long)
        self.selection_counts.index_add_(0, arms, increments)
        self.reward_sums.index_add_(0, arms, rewards)
        self.total_updates += int(arms.numel())

    def apply_policy(self, observation, actor, *policy_args):
        del actor, policy_args
        batch_size = int(observation['user_profile']['user_id'].shape[0])
        indices, ucb = self.select(batch_size)
        return {'indices': indices, 'ucb': ucb}

    def get_report(self, smoothness=10):
        """Use the A2C/LinUCB common metrics plus UCB diagnostics."""
        episode_report = self.env.get_report(smoothness)
        training_report = {
            key: np.mean(values[-smoothness:])
            for key, values in self.eval_history.items()
        }
        training_report.update({
            'ucb_alpha': self.alpha,
            'ucb_bandit_updates': self.total_updates,
            'ucb_observed_arms': int(
                torch.count_nonzero(self.selection_counts).item()
            ),
        })
        return episode_report, training_report

    def train(self):
        if len(self.n_iter) > 1:
            self.load()
        step_offset = sum(self.n_iter[:-1])
        observation = self.env.reset()
        self._setup_training_monitors()

        for step in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            with torch.no_grad():
                batch_size = int(
                    observation['user_profile']['user_id'].shape[0]
                )
                action_indices, _ = self.select(batch_size)
                observation, feedback, _, _ = self.env.step(
                    {'action': action_indices}
                )
                weights = self.env.response_weights.view(1, 1, -1)
                arm_rewards = torch.sum(
                    feedback['immediate_response'] * weights, dim=2
                )
                feedback['immediate_response_weight'] = self.env.response_weights
                reward = self.reward_func(feedback)
                self.update(action_indices, arm_rewards)
                self._record_training_step(reward, feedback)

            if step > 0 and step % self.check_episode == 0:
                self._write_training_report(step)
            if step > 0 and step % self.save_episode == 0:
                self.save()

        self.save()
        self.env.stop()

    def save(self):
        torch.save({
            'algorithm': 'context_free_ucb_semi_bandit',
            'alpha': self.alpha,
            'n_arms': self.n_arms,
            'slate_size': self.slate_size,
            'selection_counts': self.selection_counts.detach().cpu(),
            'reward_sums': self.reward_sums.detach().cpu(),
            'total_updates': self.total_updates,
        }, self.save_path + '_ucb')

    def load(self):
        checkpoint = torch.load(
            self.save_path + '_ucb', map_location=self.device
        )
        expected = (self.n_arms, self.slate_size)
        actual = (
            int(checkpoint['n_arms']), int(checkpoint['slate_size'])
        )
        if actual != expected:
            raise ValueError(
                f'Incompatible UCB checkpoint: expected {expected}, got {actual}'
            )
        self.alpha = float(checkpoint['alpha'])
        self.selection_counts = checkpoint['selection_counts'].to(
            self.device, dtype=torch.long
        )
        self.reward_sums = checkpoint['reward_sums'].to(
            self.device, dtype=torch.float32
        )
        self.total_updates = int(checkpoint.get('total_updates', 0))
