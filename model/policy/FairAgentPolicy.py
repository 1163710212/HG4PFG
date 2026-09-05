import copy
import os

import numpy as np
import pandas as pd
import torch

from model.policy.OneStageHyperPolicy_with_DotScore import OneStageHyperPolicy_with_DotScore


class FairAgentPolicy(OneStageHyperPolicy_with_DotScore):
    """Preference-aware DQN policy used by FairAgent.

    The inherited state encoder and dot-product scoring head have exactly the
    same parameter layout as ``OneStageHyperPolicy_with_DotScore``.  This lets
    FairAgent initialize its user/item representations and ranking head from a
    trained A2C actor, which is the knowledge-inheritance step in the paper.

    Actions stored by this policy are zero-based positions in the environment's
    candidate array.  Encoded item IDs in observations remain one-based.
    """

    @staticmethod
    def parse_model_args(parser):
        parser = OneStageHyperPolicy_with_DotScore.parse_model_args(parser)
        parser.add_argument(
            '--fair_backbone_actor_path', type=str, default='',
            help='A2C actor checkpoint used to initialize FairAgent embeddings and scorer',
        )
        parser.add_argument(
            '--fair_candidate_size', type=int, default=1000,
            help='size of the preference-aware action space A_u^t',
        )
        parser.add_argument(
            '--fair_backbone_ratio', type=float, default=0.3,
            help='chronological positive-interaction prefix used to define old items',
        )
        parser.add_argument(
            '--fair_history_length', type=int, default=20,
            help='number of recent positive interactions used for P_new and TGF(H_u)',
        )
        parser.add_argument(
            '--fair_default_new_ratio', type=float, default=-1.0,
            help='P_new for users without positive history; negative uses the dataset new-item ratio',
        )
        parser.add_argument(
            '--fair_positive_feedback', type=str, default='is_click',
            help='positive response used by FairAgent for history and accuracy reward',
        )
        return parser

    def __init__(self, args, environment):
        self.fair_candidate_size = args.fair_candidate_size
        self.fair_backbone_ratio = args.fair_backbone_ratio
        self.fair_history_length = args.fair_history_length
        self.fair_positive_feedback = args.fair_positive_feedback
        self.fair_default_new_ratio = args.fair_default_new_ratio

        if self.fair_candidate_size <= 0:
            raise ValueError('fair_candidate_size must be positive')
        if not 0.0 < self.fair_backbone_ratio < 1.0:
            raise ValueError('fair_backbone_ratio must be in (0, 1)')
        if self.fair_history_length <= 0:
            raise ValueError('fair_history_length must be positive')
        if self.fair_default_new_ratio > 1.0:
            raise ValueError('fair_default_new_ratio must not exceed 1')

        super().__init__(args, environment)

        new_item_mask, item_entry_order, item_introduction_stage = (
            self._build_temporal_item_metadata(environment)
        )
        self.register_buffer('new_item_mask', new_item_mask)
        self.register_buffer('item_entry_order', item_entry_order)
        self.register_buffer('item_introduction_stage', item_introduction_stage)
        self.register_buffer(
            'current_available_mask', item_introduction_stage == 0
        )
        self.register_buffer('old_item_indices', torch.nonzero(~new_item_mask, as_tuple=False).view(-1))
        self.register_buffer('new_item_indices', torch.nonzero(new_item_mask, as_tuple=False).view(-1))
        self.dynamic_stage_count = 5
        self.current_dynamic_stage = 0

        if self.old_item_indices.numel() == 0 or self.new_item_indices.numel() == 0:
            raise ValueError(
                'FairAgent requires both old and new items; check fair_backbone_ratio and timestamps'
            )
        if self.fair_candidate_size < self.slate_size:
            raise ValueError(
                f'fair_candidate_size ({self.fair_candidate_size}) must be at least '
                f'slate_size ({self.slate_size})'
            )

        available_new = new_item_mask & self.current_available_mask
        dataset_new_ratio = float(
            available_new.sum().float().item()
            / self.current_available_mask.sum().clamp(min=1).float().item()
        )
        if self.fair_default_new_ratio < 0.0:
            self.fair_default_new_ratio = dataset_new_ratio

        if args.fair_backbone_actor_path:
            self._load_backbone_actor(args.fair_backbone_actor_path)

        # Keep a frozen teacher for the paper's within-group candidate ranking.
        # The trainable copies above form Q_theta and continue to learn.
        self.backbone_user_encoder = copy.deepcopy(self.user_encoder)
        self.backbone_action_layer = copy.deepcopy(self.hyper_action_layer)
        self.backbone_user_encoder.eval()
        self.backbone_action_layer.eval()
        for parameter in self.backbone_user_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.backbone_action_layer.parameters():
            parameter.requires_grad_(False)

        print(
            'FairAgent temporal split: '
            f'old_items={self.old_item_indices.numel()}, '
            f'new_items={self.new_item_indices.numel()}, '
            f'train_available={self.current_available_mask.sum().item()}, '
            f'train_new_ratio={dataset_new_ratio:.6f}, '
            f'backbone_ratio={self.fair_backbone_ratio:.3f}'
        )

    def to(self, device):
        new_self = super().to(device)
        # OneStagePolicy.to() explicitly moves the trainable user encoders so
        # their ordinary (non-buffer) attention/position tensors follow the
        # model.  FairAgent creates this frozen teacher only after the parent
        # constructor, so it needs the same explicit migration here.
        if hasattr(new_self, 'backbone_user_encoder'):
            new_self.backbone_user_encoder.device = device
            new_self.backbone_user_encoder = (
                new_self.backbone_user_encoder.to(device)
            )
            new_self.backbone_action_layer = (
                new_self.backbone_action_layer.to(device)
            )
        return new_self

    def _build_temporal_item_metadata(self, environment):
        """Reproduce the paper's chronological old/new construction.

        The authors preprocess positive interactions chronologically, train the
        backbone on the first 30%, train FairAgent on the following 20%, and use
        the remaining half for five dynamic stages.  An item absent from the
        backbone prefix is a new item.  This must not use ``item_types.csv`` in
        this repository because that file stores popularity labels.
        """
        reader = environment.reader
        log_data = reader.log_data
        required = {'user_id', 'video_id', self.fair_positive_feedback}
        missing = sorted(required.difference(log_data.columns))
        if missing:
            raise ValueError(f'FairAgent temporal split is missing columns: {missing}')

        if 'time_ms' in log_data.columns:
            time_column = 'time_ms'
        elif 'timestamp' in log_data.columns:
            time_column = 'timestamp'
        elif 'date' in log_data.columns:
            time_column = 'date'
        else:
            raise ValueError(
                'FairAgent requires time_ms, timestamp or date to identify '
                'item entry order'
            )

        positive = log_data.loc[
            log_data[self.fair_positive_feedback] > 0,
            ['user_id', 'video_id', time_column],
        ]
        if positive.empty:
            raise ValueError(
                f'No positive rows found for feedback {self.fair_positive_feedback!r}'
            )

        # The released preprocessing keeps the last user-item interaction before
        # sorting and splitting.  Match that behavior so the 30/20/50 protocol is
        # reproducible in this runtime.
        positive = positive.drop_duplicates(['user_id', 'video_id'], keep='last')
        positive = positive.sort_values(time_column, kind='stable').reset_index(drop=True)
        backbone_end = max(1, int(len(positive) * self.fair_backbone_ratio))
        fairagent_train_end = max(
            backbone_end + 1,
            int(len(positive) * 0.5),
        )
        old_raw_items = set(positive.iloc[:backbone_end]['video_id'].tolist())

        raw_items = list(reader.items)
        new_item_mask = torch.tensor(
            [raw_item not in old_raw_items for raw_item in raw_items],
            dtype=torch.bool,
        )

        first_entry = (
            log_data[['video_id', time_column]]
            .groupby('video_id', sort=False)[time_column]
            .min()
            .to_dict()
        )
        missing_time = float('inf')
        entry_values = np.asarray(
            [first_entry.get(raw_item, missing_time) for raw_item in raw_items],
            dtype=np.float64,
        )
        stable_order = np.argsort(entry_values, kind='stable')
        entry_rank = np.empty(len(raw_items), dtype=np.int64)
        entry_rank[stable_order] = np.arange(len(raw_items), dtype=np.int64)

        # The remaining chronological 50% is split into five recommendation
        # stages in the paper.  Stage zero denotes the 30% backbone + 20%
        # FairAgent training prefix; stages one through five introduce later
        # items cumulatively.
        first_positive_position = (
            positive.reset_index()
            .groupby('video_id', sort=False)['index']
            .min()
            .to_dict()
        )
        test_size = max(len(positive) - fairagent_train_end, 1)
        stage_width = test_size / 5.0
        introduction_stage = []
        for raw_item in raw_items:
            first_position = first_positive_position.get(raw_item)
            if first_position is None:
                stage = 5
            elif first_position < fairagent_train_end:
                stage = 0
            else:
                stage = min(
                    5,
                    int((first_position - fairagent_train_end) / stage_width) + 1,
                )
            introduction_stage.append(stage)
        return (
            new_item_mask,
            torch.from_numpy(entry_rank),
            torch.tensor(introduction_stage, dtype=torch.long),
        )

    def set_dynamic_stage(self, stage):
        """Expose items available through the requested DRS stage."""
        if not 0 <= stage <= self.dynamic_stage_count:
            raise ValueError(
                f'dynamic stage must be in [0, {self.dynamic_stage_count}], got {stage}'
            )
        self.current_available_mask.copy_(self.item_introduction_stage <= stage)
        self.current_dynamic_stage = int(stage)

    def _load_backbone_actor(self, checkpoint_path):
        checkpoint_path = os.path.realpath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'FairAgent backbone actor not found: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        if not isinstance(checkpoint, dict):
            raise TypeError(f'Unsupported FairAgent backbone checkpoint: {checkpoint_path}')

        own_state = self.state_dict()
        compatible = {
            name: value
            for name, value in checkpoint.items()
            if name in own_state and own_state[name].shape == value.shape
        }
        inherited = [
            name for name in compatible
            if name.startswith('user_encoder.') or name.startswith('hyper_action_layer.')
        ]
        if not inherited:
            raise ValueError(
                'Backbone checkpoint has no compatible user encoder or ranking head: '
                f'{checkpoint_path}'
            )
        self.load_state_dict(compatible, strict=False)
        print(
            f'FairAgent inherited {len(inherited)} backbone tensors from {checkpoint_path}'
        )

    def _score_all_items(self, user_state, candidates, batch_wise=False):
        batch_size = user_state.shape[0]
        hyper_action = self.hyper_action_layer(user_state).view(batch_size, self.action_dim)
        candidate_item_enc, item_reg = self.user_encoder.get_item_encoding(
            candidates['item_id'],
            {key[3:]: value for key, value in candidates.items() if key != 'item_id'},
            batch_size if batch_wise else 1,
        )
        scores = self.get_score(hyper_action, candidate_item_enc, self.enc_dim)
        return scores, hyper_action, item_reg

    def _score_backbone_items(self, observation, candidates):
        observation = self._stage_observation(observation)
        feed_dict = {}
        feed_dict.update(observation['user_profile'])
        feed_dict.update(observation['user_history'])
        with torch.no_grad():
            user_state = self.backbone_user_encoder.get_forward(feed_dict)['state']
            batch_size = user_state.shape[0]
            hyper_action = self.backbone_action_layer(user_state).view(
                batch_size, self.action_dim
            )
            candidate_item_enc, _ = self.backbone_user_encoder.get_item_encoding(
                candidates['item_id'],
                {key[3:]: value for key, value in candidates.items() if key != 'item_id'},
                1,
            )
            return self.get_score(hyper_action, candidate_item_enc, self.enc_dim)

    def train(self, mode=True):
        super().train(mode)
        # ``nn.Module.train`` recurses into all children.  The inherited
        # backbone is a fixed teacher and must keep dropout disabled.
        if hasattr(self, 'backbone_user_encoder'):
            self.backbone_user_encoder.eval()
            self.backbone_action_layer.eval()
        return self

    def _stage_observation(self, observation):
        """Hide history items that have not entered the current DRS stage."""
        history_dict = observation['user_history']
        history = history_dict['history']
        safe_indices = torch.clamp(history - 1, min=0, max=self.item_num - 1)
        available = (history > 0) & self.current_available_mask[safe_indices]
        unavailable = (history > 0) & ~available
        if not torch.any(unavailable):
            return observation

        batch_size, history_length = history.shape
        filtered_history = dict(history_dict)
        filtered_history['history'] = history.masked_fill(unavailable, 0)
        filtered_history['history_length'] = available.sum(dim=1)
        for key, value in history_dict.items():
            if key.startswith('history_if_'):
                shaped = value.view(batch_size, history_length, -1).clone()
                shaped.masked_fill_(unavailable.unsqueeze(2), 0.0)
                filtered_history[key] = shaped.view_as(value)
            elif (
                key.startswith('history_')
                and key not in {'history_length'}
                and value.ndim == 2
                and value.shape[1] == history_length
            ):
                filtered_history[key] = value.masked_fill(unavailable, 0.0)
        return {
            'user_profile': observation['user_profile'],
            'user_history': filtered_history,
        }

    def get_user_state(self, observation):
        return super().get_user_state(self._stage_observation(observation))

    def _positive_history_mask(self, observation, history):
        history_dict = observation['user_history']
        response_key = f'history_{self.fair_positive_feedback}'
        if response_key not in history_dict:
            raise KeyError(f'Missing FairAgent history response: {response_key}')
        safe_indices = torch.clamp(history - 1, min=0, max=self.item_num - 1)
        is_available = self.current_available_mask[safe_indices]
        return (
            (history > 0)
            & is_available
            & (history_dict[response_key][:, -history.shape[1]:] > 0)
        )

    def get_history_new_ratio(self, observation):
        history = observation['user_history']['history'][:, -self.fair_history_length:]
        positive_mask = self._positive_history_mask(observation, history)
        safe_indices = torch.clamp(history - 1, min=0, max=self.item_num - 1)
        history_is_new = self.new_item_mask[safe_indices] & positive_mask
        positive_count = positive_mask.sum(dim=1)
        new_count = history_is_new.sum(dim=1)
        fallback = torch.full(
            positive_count.shape,
            float(self.fair_default_new_ratio),
            dtype=torch.float,
            device=history.device,
        )
        return torch.where(
            positive_count > 0,
            new_count.float() / positive_count.clamp(min=1).float(),
            fallback,
        ).clamp(0.0, 1.0)

    def _preference_candidate_mask(self, scores, new_ratio):
        """Construct A_u^t with an old/new quota governed by P_new.

        Within each type, the highest inherited-backbone/DQN scores are retained,
        matching the greedy within-group construction described in Section 5.2.2.
        """
        batch_size, item_count = scores.shape
        candidate_size = min(self.fair_candidate_size, item_count)
        candidate_mask = torch.zeros_like(scores, dtype=torch.bool)

        old_indices = torch.nonzero(
            self.current_available_mask & ~self.new_item_mask, as_tuple=False
        ).view(-1)
        new_indices = torch.nonzero(
            self.current_available_mask & self.new_item_mask, as_tuple=False
        ).view(-1)
        old_available = int(old_indices.numel())
        new_available = int(new_indices.numel())
        candidate_size = min(candidate_size, old_available + new_available)
        if candidate_size < self.slate_size:
            raise RuntimeError(
                f'Dynamic stage {self.current_dynamic_stage} exposes only '
                f'{old_available + new_available} items for slate_size={self.slate_size}'
            )
        for row in range(batch_size):
            wanted_new = int(torch.round(new_ratio[row] * candidate_size).item())
            n_new = min(max(wanted_new, 0), new_available)
            n_old = min(candidate_size - n_new, old_available)

            shortfall = candidate_size - n_new - n_old
            if shortfall > 0:
                add_new = min(shortfall, new_available - n_new)
                n_new += add_new
                shortfall -= add_new
            if shortfall > 0:
                n_old += min(shortfall, old_available - n_old)

            if n_new > 0:
                local = torch.topk(scores[row, new_indices], k=n_new).indices
                candidate_mask[row, new_indices[local]] = True
            if n_old > 0:
                local = torch.topk(scores[row, old_indices], k=n_old).indices
                candidate_mask[row, old_indices[local]] = True

        return candidate_mask

    def generate_action(self, state_dict, feed_dict):
        user_state = state_dict['state']
        candidates = feed_dict['candidates']
        epsilon = float(feed_dict['epsilon'])
        do_explore = bool(feed_dict['do_explore'])
        batch_wise = bool(feed_dict['batch_wise'])

        scores, hyper_action, item_reg = self._score_all_items(
            user_state, candidates, batch_wise=batch_wise
        )
        backbone_scores = self._score_backbone_items(
            feed_dict['observation'], candidates
        )
        new_ratio = self.get_history_new_ratio(feed_dict['observation'])
        available = self._preference_candidate_mask(backbone_scores, new_ratio)

        selected = []
        batch_size = scores.shape[0]
        batch_rows = torch.arange(batch_size, device=scores.device)
        for _ in range(self.slate_size):
            greedy_scores = scores.masked_fill(~available, -torch.inf)
            greedy = torch.argmax(greedy_scores, dim=1)
            if do_explore and epsilon > 0.0:
                random_choice = torch.multinomial(available.float(), 1).squeeze(1)
                use_random = torch.rand(batch_size, device=scores.device) < epsilon
                choice = torch.where(use_random, random_choice, greedy)
            else:
                choice = greedy
            selected.append(choice)
            available[batch_rows, choice] = False

        indices = torch.stack(selected, dim=1)
        action_scores = torch.gather(scores, 1, indices)
        if batch_wise:
            effect_action = torch.gather(candidates['item_id'], 1, indices)
        else:
            effect_action = candidates['item_id'][indices]

        reg = item_reg + self.get_regularization(self.hyper_action_layer)
        return {
            'preds': action_scores,
            'action': indices,
            'indices': indices,
            'hyper_action': hyper_action,
            'effect_action': effect_action,
            'all_preds': scores,
            'new_item_ratio': new_ratio,
            'reg': reg,
        }

    def evaluate_actions(self, observation, action_indices, candidates):
        state_dict = self.get_user_state(observation)
        scores, _, _ = self._score_all_items(state_dict['state'], candidates, batch_wise=False)
        return torch.gather(scores, 1, action_indices).sum(dim=1)

    def compute_tgf(self, item_indices, valid_mask=None):
        """Compute the paper's time-aware group fairness for each row.

        Old items are weighted from ``n_old`` down to one in entry order.  New
        items are weighted from one up to ``n_old``.  Exposure follows
        ``1/log2(rank+1)`` and ranks are compacted over valid items.
        """
        if item_indices.ndim != 2:
            raise ValueError('item_indices must have shape (batch, list_length)')
        if valid_mask is None:
            valid_mask = torch.ones_like(item_indices, dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()

        safe_indices = item_indices.clamp(min=0, max=self.item_num - 1)
        is_new = self.new_item_mask[safe_indices]
        entry_order = self.item_entry_order[safe_indices]
        batch_size, list_length = item_indices.shape
        positions = torch.arange(list_length, device=item_indices.device)

        # Entry rank within the selected old/new group.  Stable position order
        # breaks ties deterministically.
        left_entry = entry_order.unsqueeze(2)
        right_entry = entry_order.unsqueeze(1)
        left_pos = positions.view(1, list_length, 1)
        right_pos = positions.view(1, 1, list_length)
        right_is_older = (right_entry < left_entry) | (
            (right_entry == left_entry) & (right_pos < left_pos)
        )
        same_group = is_new.unsqueeze(2) == is_new.unsqueeze(1)
        valid_other = valid_mask.unsqueeze(1)
        entry_rank = 1 + (right_is_older & same_group & valid_other).sum(dim=2)

        old_mask = valid_mask & ~is_new
        new_mask = valid_mask & is_new
        old_count = old_mask.sum(dim=1)
        new_count = new_mask.sum(dim=1)

        old_weight = old_count.unsqueeze(1) + 1 - entry_rank
        new_denominator = (new_count - 1).clamp(min=1).unsqueeze(1).float()
        new_weight = 1.0 + (
            (entry_rank - 1).float()
            * (old_count - 1).clamp(min=0).unsqueeze(1).float()
            / new_denominator
        )
        new_weight = torch.where(
            (new_count == 1).unsqueeze(1),
            torch.ones_like(new_weight),
            new_weight,
        )

        compact_rank = valid_mask.long().cumsum(dim=1).clamp(min=1).float()
        exposure = 1.0 / torch.log2(compact_rank + 1.0)
        old_part = (
            (exposure * old_weight.float() * old_mask.float()).sum(dim=1)
            / old_count.clamp(min=1).float()
        )
        new_part = (
            (exposure * new_weight * new_mask.float()).sum(dim=1)
            / new_count.clamp(min=1).float()
        )
        return old_part - new_part

    def compute_history_tgf(self, observation):
        history = observation['user_history']['history'][:, -self.fair_history_length:]
        valid_mask = self._positive_history_mask(observation, history)
        item_indices = torch.clamp(history - 1, min=0, max=self.item_num - 1)
        return self.compute_tgf(item_indices, valid_mask)

    def compute_prefix_tgf(self, action_indices):
        batch_size, slate_size = action_indices.shape
        prefix_values = [torch.zeros(batch_size, device=action_indices.device)]
        positions = torch.arange(slate_size, device=action_indices.device).view(1, -1)
        for prefix_length in range(1, slate_size + 1):
            valid_mask = positions < prefix_length
            valid_mask = valid_mask.expand(batch_size, -1)
            prefix_values.append(self.compute_tgf(action_indices, valid_mask))
        return torch.stack(prefix_values, dim=1)
