"""Disjoint LinUCB for the whole-session recommendation simulator.

This implements Algorithm 1 from Li et al., *A Contextual-Bandit
Approach to Personalized News Article Recommendation*.  A recommendation
slate is handled with the standard semi-bandit extension: select the K arms
with the largest UCBs and update every selected arm from its own observed
feedback.
"""

import hashlib
import math
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import model.reward as reward_functions
from model.agent.BaseRLAgent import BaseRLAgent


class HashedUserContext:
    """Build a compact, normalized user context without fitting offline data.

    The paper reduces a large sparse user vector to five preference features
    and appends a constant.  The datasets here do not provide that fitted
    projection, so a deterministic signed feature hash provides the same
    low-dimensional contract without looking at test feedback.  Static user
    metadata is augmented with causal history-response rates and the current
    popularity preference.
    """

    def __init__(self, context_dim, response_types, max_history_length, device):
        if context_dim < 2:
            raise ValueError("linucb_context_dim must be at least 2")
        self.context_dim = context_dim
        self.response_types = tuple(response_types)
        self.max_history_length = max(1, int(max_history_length))
        self.device = torch.device(device)
        self.schema = None
        self.bucket_indices = None
        self.bucket_signs = None

    @staticmethod
    def _flatten(value, batch_size):
        return value.reshape(batch_size, -1).to(dtype=torch.float32)

    def _raw_features(self, observation):
        profile = observation['user_profile']
        history = observation['user_history']
        batch_size = int(profile['user_id'].shape[0])
        parts = []
        schema = []

        # Deliberately exclude user_id: LinUCB should generalize through user
        # attributes instead of learning an identity lookup table.
        for key in sorted(k for k in profile if k.startswith('uf_')):
            value = self._flatten(profile[key], batch_size)
            parts.append(value)
            schema.append((key, value.shape[1]))

        history_length = history.get('history_length')
        if history_length is None:
            length = torch.zeros(batch_size, 1, device=self.device)
        else:
            length = self._flatten(history_length, batch_size)[:, :1]
        effective_length = length.clamp(min=1.0, max=float(self.max_history_length))
        normalized_length = length.clamp(
            min=0.0, max=float(self.max_history_length)
        ) / float(self.max_history_length)
        parts.append(normalized_length)
        schema.append(('history_length_ratio', 1))

        popularity_preference = history.get('user_pop_prefer')
        if popularity_preference is None:
            popularity_preference = torch.zeros(batch_size, 1, device=self.device)
        else:
            popularity_preference = self._flatten(
                popularity_preference, batch_size
            )[:, :1]
        parts.append(popularity_preference)
        schema.append(('user_pop_prefer', 1))

        for response in self.response_types:
            key = 'history_' + response
            values = history.get(key)
            if values is None:
                rate = torch.zeros(batch_size, 1, device=self.device)
            else:
                values = self._flatten(values, batch_size)
                rate = values.sum(dim=1, keepdim=True) / effective_length
            parts.append(rate)
            schema.append((key + '_rate', 1))

        if not parts:
            raise ValueError("observation contains no usable LinUCB context features")
        raw = torch.cat(parts, dim=1).to(self.device)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return raw, tuple(schema)

    def _initialize_projection(self, schema):
        buckets = []
        signs = []
        n_buckets = self.context_dim - 1
        for name, width in schema:
            for index in range(width):
                digest = hashlib.sha256(
                    f'{name}:{index}'.encode('utf-8')
                ).digest()
                buckets.append(int.from_bytes(digest[:8], 'little') % n_buckets)
                signs.append(1.0 if digest[8] & 1 else -1.0)
        self.schema = schema
        self.bucket_indices = torch.tensor(
            buckets, dtype=torch.long, device=self.device
        )
        self.bucket_signs = torch.tensor(
            signs, dtype=torch.float32, device=self.device
        )

    def encode(self, observation, dtype):
        raw, schema = self._raw_features(observation)
        if self.schema is None:
            self._initialize_projection(schema)
        elif schema != self.schema:
            raise ValueError(
                "Train/test LinUCB context schemas differ: "
                f"train={self.schema!r}, current={schema!r}"
            )

        projected = torch.zeros(
            raw.shape[0], self.context_dim - 1,
            device=self.device, dtype=torch.float32,
        )
        projected.scatter_add_(
            1,
            self.bucket_indices.view(1, -1).expand(raw.shape[0], -1),
            raw * self.bucket_signs.view(1, -1),
        )
        norm = torch.linalg.vector_norm(projected, dim=1, keepdim=True)
        projected = projected / norm.clamp_min(1.0e-12)
        bias = torch.ones(raw.shape[0], 1, device=self.device)
        return torch.cat((bias, projected), dim=1).to(dtype=dtype)

    def state_dict(self):
        return {
            'schema': self.schema,
            'bucket_indices': (
                None if self.bucket_indices is None
                else self.bucket_indices.detach().cpu()
            ),
            'bucket_signs': (
                None if self.bucket_signs is None
                else self.bucket_signs.detach().cpu()
            ),
        }

    def load_state_dict(self, state):
        schema = state.get('schema')
        self.schema = None if schema is None else tuple(
            (name, int(width)) for name, width in schema
        )
        bucket_indices = state.get('bucket_indices')
        bucket_signs = state.get('bucket_signs')
        self.bucket_indices = (
            None if bucket_indices is None
            else bucket_indices.to(self.device, dtype=torch.long)
        )
        self.bucket_signs = (
            None if bucket_signs is None
            else bucket_signs.to(self.device, dtype=torch.float32)
        )


class LinUCB(BaseRLAgent):
    """Disjoint linear UCB with batched semi-bandit slate updates."""

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
            '--linucb_alpha', type=float, default=1.0,
            help='UCB exploration coefficient alpha',
        )
        parser.add_argument(
            '--linucb_ridge', type=float, default=1.0,
            help='positive ridge coefficient used to initialize every A_a',
        )
        parser.add_argument(
            '--linucb_context_dim', type=int, default=6,
            help='context size including the constant feature',
        )
        parser.add_argument(
            '--linucb_score_chunk_size', type=int, default=4096,
            help='number of arms scored at once',
        )
        return parser

    def __init__(self, args, env):
        if not math.isfinite(args.linucb_alpha) or args.linucb_alpha < 0.0:
            raise ValueError("linucb_alpha must be finite and non-negative")
        if not math.isfinite(args.linucb_ridge) or args.linucb_ridge <= 0.0:
            raise ValueError("linucb_ridge must be finite and positive")
        if args.linucb_score_chunk_size <= 0:
            raise ValueError("linucb_score_chunk_size must be positive")
        if args.linucb_context_dim < 2:
            raise ValueError("linucb_context_dim must be at least 2")
        if args.check_episode <= 0 or args.save_episode <= 0:
            raise ValueError("check_episode and save_episode must be positive")
        if not args.n_iter or any(iterations <= 0 for iterations in args.n_iter):
            raise ValueError("every n_iter stage must be positive")

        self.device = torch.device(args.device)
        self.env = env
        self.actor = nn.Identity().to(self.device)
        self.reward_func = getattr(reward_functions, args.reward_func)
        self.n_iter = list(args.n_iter)
        self.check_episode = args.check_episode
        self.save_episode = args.save_episode
        self.save_path = args.save_path
        self.test_n_step = args.test_n_step
        self.test_seed = args.test_seed
        self.test_repeat = args.test_repeat
        self.test_num_users = args.test_num_users
        self.alpha = float(args.linucb_alpha)
        self.ridge = float(args.linucb_ridge)
        self.context_dim = int(args.linucb_context_dim)
        self.score_chunk_size = int(args.linucb_score_chunk_size)
        self.dtype = torch.float32
        self.n_arms = int(env.n_candidate)
        self.slate_size = int(env.slate_size)
        if self.slate_size > self.n_arms:
            raise ValueError("slate_size cannot exceed the number of arms")

        identity = torch.eye(
            self.context_dim, device=self.device, dtype=self.dtype
        )
        self.A = identity.unsqueeze(0).repeat(self.n_arms, 1, 1) * self.ridge
        self.A_inv = identity.unsqueeze(0).repeat(self.n_arms, 1, 1) / self.ridge
        self.b = torch.zeros(
            self.n_arms, self.context_dim,
            device=self.device, dtype=self.dtype,
        )
        self.context_encoder = HashedUserContext(
            self.context_dim,
            env.response_types,
            env.reader.get_statistics()['max_seq_len'],
            self.device,
        )
        self.total_updates = 0

        os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
        if len(self.n_iter) == 1:
            with open(self.save_path + '.report', 'w', encoding='utf-8') as report:
                report.write(str(args) + '\n')

    def _contexts(self, observation):
        return self.context_encoder.encode(observation, self.dtype)

    def score(self, contexts, arm_indices=None):
        """Compute Eq. (5): x^T theta_a + alpha sqrt(x^T A_a^-1 x)."""
        if arm_indices is None:
            inverse = self.A_inv
            rhs = self.b
        else:
            inverse = self.A_inv[arm_indices]
            rhs = self.b[arm_indices]
        theta = torch.einsum('aij,aj->ai', inverse, rhs)
        mean = torch.matmul(contexts, theta.transpose(0, 1))
        variance = torch.einsum(
            'bd,ade,be->ba', contexts, inverse, contexts
        ).clamp_min(0.0)
        return mean + self.alpha * torch.sqrt(variance)

    def select(self, contexts):
        """Return global top-K arms while bounding temporary score memory."""
        batch_size = contexts.shape[0]
        best_scores = torch.empty(
            batch_size, 0, device=self.device, dtype=self.dtype
        )
        best_indices = torch.empty(
            batch_size, 0, device=self.device, dtype=torch.long
        )
        for start in range(0, self.n_arms, self.score_chunk_size):
            stop = min(start + self.score_chunk_size, self.n_arms)
            arms = torch.arange(start, stop, device=self.device)
            scores = self.score(contexts, arms)
            indices = arms.view(1, -1).expand(batch_size, -1)
            merged_scores = torch.cat((best_scores, scores), dim=1)
            merged_indices = torch.cat((best_indices, indices), dim=1)
            keep = min(self.slate_size, merged_scores.shape[1])
            best_scores, positions = torch.topk(
                merged_scores, k=keep, dim=1, sorted=True
            )
            best_indices = torch.gather(merged_indices, 1, positions)
        return best_indices, best_scores

    def update(self, contexts, action_indices, arm_rewards):
        """Apply A_a += xx^T and b_a += r*x for all observed slate arms."""
        if action_indices.shape != arm_rewards.shape:
            raise ValueError("action_indices and arm_rewards must have equal shape")
        batch_size, slate_size = action_indices.shape
        expanded_contexts = contexts[:, None, :].expand(
            batch_size, slate_size, self.context_dim
        ).reshape(-1, self.context_dim)
        arms = action_indices.reshape(-1)
        rewards = arm_rewards.reshape(-1).to(self.dtype)
        outer_products = torch.einsum(
            'bi,bj->bij', expanded_contexts, expanded_contexts
        )
        self.A.index_add_(0, arms, outer_products)
        self.b.index_add_(0, arms, rewards[:, None] * expanded_contexts)

        affected_arms = torch.unique(arms)
        affected_A = self.A[affected_arms]
        affected_A = 0.5 * (
            affected_A + affected_A.transpose(1, 2)
        )
        self.A[affected_arms] = affected_A
        self.A_inv[affected_arms] = torch.linalg.inv(affected_A)
        self.total_updates += int(arms.numel())

    def apply_policy(self, observation, actor, *policy_args):
        del actor, policy_args
        contexts = self._contexts(observation)
        indices, ucb = self.select(contexts)
        return {'indices': indices, 'ucb': ucb, 'context': contexts}

    def _setup_training_monitors(self):
        """Use the same online metric definitions and initial state as A2C."""
        self.eval_history = {
            'avg_reward': [],
            'reward_variance': [],
            'avg_total_reward': [0.0],
            'max_total_reward': [0.0],
            'min_total_reward': [0.0],
        }
        self.eval_history.update({
            f'{response}_rate': [] for response in self.env.response_types
        })
        self.current_sum_reward = torch.zeros(
            self.env.episode_batch_size,
            dtype=torch.float32,
            device=self.device,
        )

    def _record_training_step(self, reward, feedback):
        """Record rewards, completed returns and response rates like A2C."""
        self.current_sum_reward = self.current_sum_reward + reward
        done_mask = feedback['done'].bool()
        if torch.any(done_mask):
            completed = self.current_sum_reward[done_mask]
            self.eval_history['avg_total_reward'].append(
                completed.mean().item()
            )
            self.eval_history['max_total_reward'].append(
                completed.max().item()
            )
            self.eval_history['min_total_reward'].append(
                completed.min().item()
            )
            self.current_sum_reward[done_mask] = 0.0

        self.eval_history['avg_reward'].append(reward.mean().item())
        self.eval_history['reward_variance'].append(torch.var(reward).item())
        for response_index, response in enumerate(self.env.response_types):
            response_rate = feedback['immediate_response'][
                :, :, response_index
            ].mean().item()
            self.eval_history[f'{response}_rate'].append(response_rate)

    @staticmethod
    def _as_report_float(value):
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
        return float(value)

    def get_report(self, smoothness=10):
        """Return the same two report sections produced by BaseRLAgent."""
        episode_report = self.env.get_report(smoothness)
        training_report = {
            key: np.mean(values[-smoothness:])
            for key, values in self.eval_history.items()
        }
        # Algorithm-specific diagnostics are additive; the common metric names
        # and definitions above remain exactly those used by A2C.
        training_report.update({
            'linucb_alpha': self.alpha,
            'linucb_bandit_updates': self.total_updates,
        })
        return episode_report, training_report

    def _write_training_report(self, step):
        episode_report, training_report = self.get_report(self.check_episode)
        episode_report = {
            key: self._as_report_float(value)
            for key, value in episode_report.items()
        }
        training_report = {
            key: self._as_report_float(value)
            for key, value in training_report.items()
        }
        line = (
            f"step: {step} @ online episode: {episode_report} "
            f"@ training: {training_report}\n"
        )
        with open(self.save_path + '.report', 'a', encoding='utf-8') as outfile:
            outfile.write(line)
        print(line, end='')

    def train(self):
        if len(self.n_iter) > 1:
            self.load()
        step_offset = sum(self.n_iter[:-1])
        observation = self.env.reset()
        self._setup_training_monitors()

        for step in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            with torch.no_grad():
                contexts = self._contexts(observation)
                action_indices, _ = self.select(contexts)
                observation, feedback, _, _ = self.env.step(
                    {'action': action_indices}
                )
                weights = self.env.response_weights.view(1, 1, -1)
                arm_rewards = torch.sum(
                    feedback['immediate_response'] * weights, dim=2
                )
                feedback['immediate_response_weight'] = self.env.response_weights
                reward = self.reward_func(feedback)
                self.update(contexts, action_indices, arm_rewards)
                self._record_training_step(reward, feedback)

            if step > 0 and step % self.check_episode == 0:
                self._write_training_report(step)
            if step > 0 and step % self.save_episode == 0:
                self.save()

        self.save()
        self.env.stop()

    def save(self):
        checkpoint = {
            'algorithm': 'disjoint_linucb_semi_bandit',
            'alpha': self.alpha,
            'ridge': self.ridge,
            'context_dim': self.context_dim,
            'n_arms': self.n_arms,
            'slate_size': self.slate_size,
            'A': self.A.detach().cpu(),
            'A_inv': self.A_inv.detach().cpu(),
            'b': self.b.detach().cpu(),
            'context_encoder': self.context_encoder.state_dict(),
            'total_updates': self.total_updates,
        }
        torch.save(checkpoint, self.save_path + '_linucb')

    def load(self):
        checkpoint = torch.load(
            self.save_path + '_linucb', map_location=self.device
        )
        expected = (self.n_arms, self.context_dim, self.slate_size)
        actual = (
            int(checkpoint['n_arms']), int(checkpoint['context_dim']),
            int(checkpoint['slate_size']),
        )
        if actual != expected:
            raise ValueError(
                f"Incompatible LinUCB checkpoint: expected {expected}, got {actual}"
            )
        self.alpha = float(checkpoint['alpha'])
        self.ridge = float(checkpoint['ridge'])
        self.A = checkpoint['A'].to(self.device, dtype=self.dtype)
        self.A_inv = checkpoint['A_inv'].to(self.device, dtype=self.dtype)
        self.b = checkpoint['b'].to(self.device, dtype=self.dtype)
        self.context_encoder.load_state_dict(checkpoint['context_encoder'])
        self.total_updates = int(checkpoint.get('total_updates', 0))
