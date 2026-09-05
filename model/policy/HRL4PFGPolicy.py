import inspect
import math
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal

from model.components import DNN
from model.general import BaseModel
from model.policy.BackboneUserEncoder import BackboneUserEncoder


_MHA_SUPPORTS_IS_CAUSAL = (
    'is_causal' in inspect.signature(nn.MultiheadAttention.forward).parameters
)


def continuous_tail_coordinates(popularity_percentile, item_types):
    """Map popularity ranks to a continuous, nearly gap-free tail axis.

    Popular items occupy ``[0, 0.5)`` and long-tail items occupy ``(0.5, 1]``.
    The binary labels locate the catalogue boundary only; ordering inside each
    group continues to follow the aligned popularity percentile.
    """
    types = item_types.reshape(-1).float()
    percentile = popularity_percentile.reshape(-1).to(
        device=types.device, dtype=torch.float32
    )
    if percentile.numel() != types.numel():
        raise ValueError('popularity percentile and item type sizes differ')
    if not torch.any(types == 0) or not torch.any(types == 1):
        raise ValueError('continuous tail coordinates require both item groups')
    popular_floor = percentile[types == 1].min()
    tail_ceiling = percentile[types == 0].max()
    if not bool(popular_floor > tail_ceiling):
        raise ValueError(
            'item types are not separable by continuous popularity percentile'
        )
    catalogue_floor = percentile.min()
    popular_coordinate = 0.499 * (1.0 - percentile) / (
        1.0 - popular_floor
    ).clamp_min(1e-6)
    tail_coordinate = 0.501 + 0.499 * (
        tail_ceiling - percentile
    ) / (tail_ceiling - catalogue_floor).clamp_min(1e-6)
    return torch.where(
        types == 1, popular_coordinate, tail_coordinate
    ).clamp(0.0, 1.0)


class _SafeTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Use the documented attention path across the supported Torch versions."""

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # A value-identical view avoids the buggy fused-mask path in Torch 1.12.
        key_value = x.view_as(x)
        kwargs = {
            'attn_mask': attn_mask,
            'key_padding_mask': key_padding_mask,
            'need_weights': False,
        }
        if _MHA_SUPPORTS_IS_CAUSAL:
            kwargs['is_causal'] = is_causal
        return self.dropout1(self.self_attn(x, key_value, key_value, **kwargs)[0])


def enable_safe_transformer_attention(module):
    """Select a version-compatible Transformer attention implementation."""
    for child in module.modules():
        if isinstance(child, nn.TransformerEncoderLayer):
            child.__class__ = _SafeTransformerEncoderLayer
            if hasattr(child, 'activation_relu_or_gelu'):
                child.activation_relu_or_gelu = 0


class HRL4PFGPolicy(BaseModel):
    """Paper-aligned macro/micro policy for HRL4PFG.

    The macro actor samples a continuous prototype from a Gaussian and maps it
    to an exact catalogue item.  ``catalog_nearest`` preserves historical v5
    behavior; the opt-in modes add long-tail or v4-style gradual popularity
    constraints.  The micro actor conditions the full-item distribution on
    the preference concatenated with that mapped target.
    """

    @staticmethod
    def parse_model_args(parser):
        parser = BackboneUserEncoder.parse_model_args(parser)
        parser.add_argument(
            '--policy_action_hidden', type=int, nargs='+', default=[256, 64],
            help='hidden dimensions of the HRL actor heads',
        )
        parser.add_argument(
            '--action_std_init', type=float, default=0.1,
            help='initial standard deviation of the Gaussian target actor',
        )
        parser.add_argument(
            '--goal_std_min', type=float, default=0.02,
            help='minimum standard deviation of each target dimension',
        )
        parser.add_argument(
            '--goal_std_max', type=float, default=0.1,
            help='maximum standard deviation of each target dimension',
        )
        parser.add_argument(
            '--goal_residual_scale', type=float, default=2.0,
            help=(
                'soft-popularity mode: maximum learned macro residual around '
                'the current preference'
            ),
        )
        parser.add_argument(
            '--target_candidate_size', type=int, default=128,
            help='number of preference-compatible target items considered',
        )
        parser.add_argument(
            '--target_projection_mode',
            choices=('catalog_nearest', 'hard_tail', 'soft_popularity'),
            default='catalog_nearest',
            help=(
                'catalog_nearest preserves the v5 complete-catalogue mapping; '
                'hard_tail uses compatible tail items; soft_popularity applies '
                'the v4 compatible and gradual popularity projection'
            ),
        )
        parser.add_argument(
            '--disable_target_snapping', action='store_true',
            help=(
                'ablation: use the continuous macro prototype directly as '
                'the low-level target instead of projecting it to an item'
            ),
        )
        parser.add_argument(
            '--target_tailness_candidate_size', type=int, default=0,
            help=(
                'hard-tail mode: optionally retain this many most tailward '
                'items after preference compatibility screening'
            ),
        )
        parser.add_argument(
            '--soft_target_tail_advance', type=float, default=0.2,
            help='soft-popularity target advance beyond the current tail state',
        )
        parser.add_argument(
            '--soft_target_popularity_coef', type=float, default=4.0,
            help='continuous target-tail error weight in soft-popularity mode',
        )
        return parser

    def __init__(self, args, environment):
        self.slate_size = environment.slate_size
        self.dataset_dir = args.dataset_dir
        self.action_std_init = args.action_std_init
        self.goal_std_min = args.goal_std_min
        self.goal_std_max = args.goal_std_max
        self.goal_residual_scale = float(
            getattr(args, 'goal_residual_scale', 2.0)
        )
        self.target_candidate_size = int(
            getattr(args, 'target_candidate_size', 128)
        )
        self.target_projection_mode = getattr(
            args, 'target_projection_mode', 'catalog_nearest'
        )
        self.disable_target_snapping = bool(
            getattr(args, 'disable_target_snapping', False)
        )
        self.target_tailness_candidate_size = int(
            getattr(args, 'target_tailness_candidate_size', 0)
        )
        self.soft_target_tail_advance = float(
            getattr(args, 'soft_target_tail_advance', 0.2)
        )
        self.soft_target_popularity_coef = float(
            getattr(args, 'soft_target_popularity_coef', 4.0)
        )
        if not 0.0 < self.goal_std_min <= self.action_std_init <= self.goal_std_max:
            raise ValueError(
                'goal standard deviations must satisfy '
                '0 < goal_std_min <= action_std_init <= goal_std_max'
            )
        self.goal_log_std_min = math.log(self.goal_std_min)
        self.goal_log_std_max = math.log(self.goal_std_max)
        super().__init__(args, environment.reader.get_statistics(), args.device)
        self.display_name = 'HRL4PFGPolicy'

        if self.target_projection_mode not in (
            'catalog_nearest', 'hard_tail', 'soft_popularity'
        ):
            raise ValueError('unsupported target_projection_mode')
        if self.target_projection_mode != 'catalog_nearest':
            if self.goal_residual_scale <= 0.0:
                raise ValueError('goal_residual_scale must be positive')
            if not 0 < self.target_candidate_size <= self.item_num:
                raise ValueError(
                    'target_candidate_size must be in [1, item_num]'
                )
            if not 0 <= self.target_tailness_candidate_size <= (
                self.target_candidate_size
            ):
                raise ValueError(
                    'target_tailness_candidate_size must be in '
                    '[0, target_candidate_size]'
                )
        if self.target_projection_mode == 'soft_popularity':
            if not 0.0 <= self.soft_target_tail_advance <= 1.0:
                raise ValueError(
                    'soft_target_tail_advance must be in [0, 1]'
                )
            if self.soft_target_popularity_coef < 0.0:
                raise ValueError(
                    'soft_target_popularity_coef must be non-negative'
                )

        item_types = pd.read_csv(
            os.path.join(self.dataset_dir, 'item_types.csv')
        ).to_numpy().reshape(-1)
        if len(item_types) != self.item_num:
            raise ValueError(
                'item_types.csv has {0} rows but the environment exposes {1} items'.format(
                    len(item_types), self.item_num
                )
            )
        if not set(pd.unique(item_types)).issubset({0, 1}):
            raise ValueError('item_types.csv must be binary (1=popular, 0=long-tail)')
        self.register_buffer(
            'item_types', torch.as_tensor(item_types, dtype=torch.float32)
        )
        if self.target_projection_mode == 'soft_popularity':
            alignment = getattr(environment, 'item_metadata_alignment', {})
            if alignment.get('alignment_mode') != 'sorted_raw_item_id':
                raise ValueError(
                    'soft_popularity requires --align_item_metadata so '
                    'popularity coordinates match encoded catalogue indices'
                )
            aligned_types = environment.item_types.reshape(-1).float()
            self.item_types.copy_(aligned_types.to(self.item_types.device))
        item_tail_coordinate = (
            continuous_tail_coordinates(
                environment.item_popularity_percentile,
                self.item_types,
            )
            if self.target_projection_mode == 'soft_popularity'
            else torch.zeros_like(self.item_types)
        )
        self.register_buffer(
            'item_tail_coordinate', item_tail_coordinate, persistent=False
        )
        if self.item_num <= 1:
            raise ValueError('entropy-normalized fairness requires at least 2 items')
        if not torch.any(self.item_types == 0):
            raise ValueError('item_types.csv must contain at least one long-tail item')

        # The same frozen simulator is used by the existing HPPO path.  This
        # compatibility hook only changes which equivalent attention kernel is
        # selected on older Torch releases.
        enable_safe_transformer_attention(environment.immediate_response_model)

    def _define_params(self, args, reader_stats):
        # Independent attention state trackers serve the macro and micro levels.
        self.h_user_encoder = BackboneUserEncoder(args, reader_stats, has_up=False)
        self.l_user_encoder = BackboneUserEncoder(args, reader_stats, has_up=False)
        enable_safe_transformer_attention(self.h_user_encoder)
        enable_safe_transformer_attention(self.l_user_encoder)

        self.state_dim = self.h_user_encoder.state_dim
        self.goal_dim = self.l_user_encoder.enc_dim
        self.item_num = reader_stats['n_item']
        self.action_dim = self.slate_size
        self.effect_action_dim = self.slate_size
        self.h_hyper_action_dim = self.goal_dim
        self.h_action_dim = self.goal_dim
        self.l_action_dim = self.item_num

        # State-dependent Gaussian mean and variance for the target prototype.
        self.h_action_layer = DNN(
            self.state_dim,
            args.policy_action_hidden,
            self.goal_dim,
            dropout_rate=args.state_dropout_rate,
            do_batch_norm=True,
        )
        mean_output = self.h_action_layer.layers[-1]
        nn.init.zeros_(mean_output.weight)
        nn.init.zeros_(mean_output.bias)

        self.h_action_log_std_layer = DNN(
            self.state_dim,
            args.policy_action_hidden,
            self.goal_dim,
            dropout_rate=args.state_dropout_rate,
            do_batch_norm=True,
        )
        std_output = self.h_action_log_std_layer.layers[-1]
        nn.init.zeros_(std_output.weight)
        nn.init.constant_(std_output.bias, math.log(self.action_std_init))

        # Map the low-level attention state into the item-target space.
        self.preference_layer = DNN(
            self.l_user_encoder.state_dim,
            args.policy_action_hidden,
            self.goal_dim,
            dropout_rate=args.state_dropout_rate,
            do_batch_norm=True,
        )
        self.l_action_layer = nn.Sequential(DNN(
            2 * self.goal_dim,
            args.policy_action_hidden,
            self.item_num,
            dropout_rate=args.state_dropout_rate,
            do_batch_norm=True,
        ), nn.Softmax(dim=-1))

    def to(self, device):
        new_self = super(HRL4PFGPolicy, self).to(device)
        new_self.device = device
        # BackboneUserEncoder stores these masks as ordinary attributes.
        for encoder in (new_self.h_user_encoder, new_self.l_user_encoder):
            encoder.device = device
            encoder.attn_mask = encoder.attn_mask.to(device)
            encoder.pos_emb_getter = encoder.pos_emb_getter.to(device)
        return new_self

    def set_init_var(self):
        if self.action_std_init <= 0:
            raise ValueError('action_std_init must be positive')

    @staticmethod
    def _observation_feed(observation):
        feed_dict = {}
        feed_dict.update(observation['user_profile'])
        feed_dict.update(observation['user_history'])
        return feed_dict

    def encode_high_state(self, observation):
        return self.h_user_encoder.get_forward(
            self._observation_feed(observation)
        )['state']

    def encode_preference(self, observation):
        low_state = self.l_user_encoder.get_forward(
            self._observation_feed(observation)
        )['state']
        return self.preference_layer(low_state)

    def encode_states(self, observation):
        feed_dict = self._observation_feed(observation)
        high_output = self.h_user_encoder.get_forward(feed_dict)
        low_output = self.l_user_encoder.get_forward(feed_dict)
        preference = self.preference_layer(low_output['state'])
        return (
            high_output['state'],
            preference,
            high_output['reg'] + low_output['reg'],
        )

    def _item_encodings(self, candidates):
        item_ids = candidates['item_id'].reshape(1, -1)
        item_features = {
            key[3:]: value.reshape(1, value.shape[0], -1)
            for key, value in candidates.items()
            if key.startswith('if_')
        }
        item_encoding, _ = self.l_user_encoder.get_item_encoding(
            item_ids, item_features, 1
        )
        item_encoding = item_encoding.reshape(-1, self.goal_dim).detach()
        if item_encoding.shape[0] != self.item_num:
            raise ValueError(
                'Long-Tail Target Snapping requires the complete item set'
            )
        return item_encoding

    def _distribution(self, high_state, preference=None):
        raw_mean = self.h_action_layer(high_state)
        if getattr(self, 'target_projection_mode', 'catalog_nearest') == (
            'soft_popularity'
        ):
            if preference is None:
                raise ValueError(
                    'soft_popularity macro distribution requires preference'
                )
            mean = preference.detach() + self.goal_residual_scale * torch.tanh(
                raw_mean
            )
        else:
            # Preserve the original v5 actor exactly unless the new mode is
            # explicitly enabled.
            mean = raw_mean
        log_std = torch.clamp(
            self.h_action_log_std_layer(high_state),
            min=self.goal_log_std_min,
            max=self.goal_log_std_max,
        )
        std = log_std.exp()
        return Independent(Normal(mean, std), 1), mean, std

    def _snap_to_long_tail(
        self, prototype, candidates, preference=None, popular_preference=None,
    ):
        """Map continuous prototypes to real items under an explicit mode."""
        item_encoding = self._item_encodings(candidates)
        projection_mode = getattr(
            self, 'target_projection_mode', 'catalog_nearest'
        )
        if projection_mode == 'catalog_nearest':
            distance_sq = (
                prototype.square().sum(dim=-1, keepdim=True)
                + item_encoding.square().sum(dim=-1).reshape(1, -1)
                - 2.0 * prototype.matmul(item_encoding.transpose(0, 1))
            )
            nearest_distance_sq, nearest_index = distance_sq.min(dim=-1)
            return (
                item_encoding[nearest_index].detach(),
                nearest_distance_sq.clamp_min(0.0).sqrt(),
                nearest_index,
            )

        prototype_on_manifold = F.normalize(
            prototype, p=2.0, dim=-1
        ) * math.sqrt(self.goal_dim)
        batch_size = prototype.shape[0]
        catalogue_size = item_encoding.shape[0]
        compatible_count = min(
            int(getattr(self, 'target_candidate_size', catalogue_size)),
            catalogue_size,
        )

        if projection_mode == 'soft_popularity':
            if popular_preference is None:
                raise ValueError(
                    'soft_popularity target projection requires '
                    'popular_preference'
                )
            if preference is None:
                compatible_indices = torch.arange(
                    catalogue_size, device=item_encoding.device
                ).reshape(1, -1).expand(batch_size, -1)
            else:
                satisfaction_logits = preference.matmul(
                    item_encoding.transpose(0, 1)
                ) / float(self.goal_dim)
                compatible_indices = torch.topk(
                    satisfaction_logits, k=compatible_count, dim=-1
                ).indices
            compatible_encoding = item_encoding[compatible_indices]
            distance_sq = (
                prototype_on_manifold.square().sum(dim=-1, keepdim=True)
                + compatible_encoding.square().sum(dim=-1)
                - 2.0 * torch.sum(
                    prototype_on_manifold.unsqueeze(1) * compatible_encoding,
                    dim=-1,
                )
            ).clamp_min(0.0)
            current_tail_state = (
                1.0 - popular_preference.reshape(-1).float()
            )
            desired_tail_coordinate = (
                current_tail_state + self.soft_target_tail_advance
            ).clamp(0.0, 1.0)
            popularity_error_sq = (
                self.item_tail_coordinate[compatible_indices]
                - desired_tail_coordinate.reshape(-1, 1)
            ).square()
            score = (
                distance_sq / float(self.goal_dim)
                + self.soft_target_popularity_coef * popularity_error_sq
            )
            nearest_in_compatible = score.argmin(dim=-1)
            nearest_index = compatible_indices.gather(
                1, nearest_in_compatible.reshape(-1, 1)
            ).reshape(-1)
            nearest_distance_sq = distance_sq.gather(
                1, nearest_in_compatible.reshape(-1, 1)
            ).reshape(-1)
            return (
                item_encoding[nearest_index].detach(),
                nearest_distance_sq.sqrt(),
                nearest_index,
            )

        tail_indices = torch.where(self.item_types == 0)[0]
        tail_encoding = item_encoding[tail_indices]
        if preference is not None:
            compatible_count = min(compatible_count, tail_encoding.shape[0])
            satisfaction_logits = preference.matmul(
                tail_encoding.transpose(0, 1)
            ) / float(self.goal_dim)
            compatible_local = torch.topk(
                satisfaction_logits, k=compatible_count, dim=-1
            ).indices
            tailward_count = int(getattr(
                self, 'target_tailness_candidate_size', 0
            ))
            if tailward_count > 0:
                if not hasattr(self, 'semantic_item_tailness'):
                    raise ValueError(
                        'tailness filtering requires semantic_item_tailness'
                    )
                tailward_count = min(tailward_count, compatible_count)
                compatible_tailness = self.semantic_item_tailness[
                    tail_indices[compatible_local]
                ]
                tailward_in_compatible = torch.topk(
                    compatible_tailness, k=tailward_count, dim=-1
                ).indices
                compatible_local = compatible_local.gather(
                    1, tailward_in_compatible
                )
            compatible_encoding = tail_encoding[compatible_local]
            distance_sq = (
                prototype_on_manifold.square().sum(dim=-1, keepdim=True)
                + compatible_encoding.square().sum(dim=-1)
                - 2.0 * torch.sum(
                    prototype_on_manifold.unsqueeze(1) * compatible_encoding,
                    dim=-1,
                )
            ).clamp_min(0.0)
            nearest_distance_sq, nearest_in_compatible = distance_sq.min(dim=-1)
            nearest_local = compatible_local.gather(
                1, nearest_in_compatible.reshape(-1, 1)
            ).reshape(-1)
        else:
            distance_sq = (
                prototype_on_manifold.square().sum(dim=-1, keepdim=True)
                + tail_encoding.square().sum(dim=-1).reshape(1, -1)
                - 2.0 * prototype_on_manifold.matmul(
                    tail_encoding.transpose(0, 1)
                )
            ).clamp_min(0.0)
            nearest_distance_sq, nearest_local = distance_sq.min(dim=-1)
        nearest_index = tail_indices[nearest_local]
        return (
            item_encoding[nearest_index].detach(),
            nearest_distance_sq.sqrt(),
            nearest_index,
        )

    def _project_goal(
        self, prototype, candidates, preference=None, popular_preference=None,
    ):
        """Apply the configured macro-target ablation or item projection."""
        batch_size = prototype.shape[0]
        if getattr(self, 'disable_target_snapping', False):
            return (
                prototype.detach(),
                prototype.new_zeros(batch_size),
                torch.full(
                    (batch_size,), -1, dtype=torch.long,
                    device=prototype.device,
                ),
            )
        return self._snap_to_long_tail(
            prototype,
            candidates,
            preference,
            popular_preference,
        )

    def deterministic_snapped_goal(self, observation, candidates):
        """Return the configured target for the deterministic policy mean."""
        high_state = self.encode_high_state(observation)
        preference = (
            self.encode_preference(observation)
            if self.target_projection_mode == 'soft_popularity'
            else None
        )
        _, mean, _ = self._distribution(high_state, preference)
        goal, _, _ = self._project_goal(
            mean,
            candidates,
            preference,
            (
                observation['user_history']['user_pop_prefer']
                if self.target_projection_mode == 'soft_popularity'
                else None
            ),
        )
        return goal

    def _low_distribution(self, low_state):
        item_probs = self.l_action_layer(low_state)
        return Categorical(probs=item_probs), item_probs

    def forward(self, feed_dict, return_prob=True):
        observation = feed_dict['observation']
        high_state, preference, reg = self.encode_states(observation)
        high_dist, high_mean, high_std = self._distribution(
            high_state, preference
        )

        sampled_prototype = (
            high_dist.sample() if feed_dict['do_explore'] else high_mean
        )
        new_goal_mask = feed_dict['new_goal_mask'].bool().reshape(-1)
        goal = feed_dict['previous_goal'].clone()
        goal_snap_distance = feed_dict['previous_goal_snap_distance'].clone()
        goal_index = torch.full(
            (preference.shape[0],), -1, dtype=torch.long,
            device=preference.device,
        )
        if torch.any(new_goal_mask):
            sampled_goal, sampled_snap_distance, sampled_goal_index = (
                self._project_goal(
                    sampled_prototype[new_goal_mask],
                    feed_dict['candidates'],
                    preference[new_goal_mask],
                    observation['user_history']['user_pop_prefer'][
                        new_goal_mask
                    ],
                )
            )
            goal[new_goal_mask] = sampled_goal
            goal_snap_distance[new_goal_mask] = sampled_snap_distance
            goal_index[new_goal_mask] = sampled_goal_index

        low_state = torch.cat((preference, goal), dim=-1)
        low_dist, item_probs = self._low_distribution(low_state)
        fairness_reward = -torch.sum(
            item_probs * torch.log(item_probs + 1e-12), dim=-1
        ) / math.log(self.item_num)
        if feed_dict['do_explore']:
            action = torch.multinomial(
                item_probs, num_samples=self.slate_size, replacement=False
            )
        else:
            action = torch.topk(item_probs, k=self.slate_size, dim=1).indices
        l_action_log_prob = torch.mean(low_dist.log_prob(action.transpose(1, 0)).transpose(1, 0), dim=1).squeeze()

        return {
            'h_state': high_state,
            'preference': preference,
            'l_state': low_state,
            'goal': goal,
            'goal_index': goal_index,
            'goal_snap_distance': goal_snap_distance,
            'goal_std': high_std.mean(dim=-1),
            'h_action': sampled_prototype,
            'h_action_log_prob': high_dist.log_prob(sampled_prototype),
            'l_action': action,
            'l_action_log_prob': l_action_log_prob,
            'l_entropy': low_dist.entropy(),
            'fairness_reward': fairness_reward,
            'indices': action,
            'effect_action': action,
            'unpopular_item_ratio': (1.0 - self.item_types[action]).mean(dim=-1),
            'reg': reg + self.get_regularization(
                self.h_action_layer,
                self.h_action_log_std_layer,
                self.preference_layer,
                self.l_action_layer,
            ),
        }

    def evaluate_high(self, observation, prototype):
        high_state = self.encode_high_state(observation)
        preference = (
            self.encode_preference(observation)
            if self.target_projection_mode == 'soft_popularity'
            else None
        )
        distribution, _, _ = self._distribution(high_state, preference)
        return (
            distribution.log_prob(prototype),
            distribution.entropy(),
            high_state,
        )

    def evaluate_low(self, low_state, action, detach_state=True):
        """Evaluate a rollout action from its cached low-level state.

        HPPO evaluates its micro policy from the state stored during rollout,
        so PPO updates only the low-level action head instead of rerunning the
        user encoder through the current actor.  Preserve the same separation
        here for a controlled lower-level comparison.
        """
        if detach_state:
            low_state = low_state.detach()
        distribution, _ = self._low_distribution(low_state)
        l_action_log_prob = torch.mean(distribution.log_prob(action.transpose(1, 0)).transpose(1, 0), dim=1).squeeze()
        return (
            l_action_log_prob,
            distribution.entropy(),
            low_state,
        )
