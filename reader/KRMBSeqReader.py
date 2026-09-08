import math
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from reader.BaseReader import BaseReader
from utils import (
    get_multihot_vocab,
    get_onehot_vocab,
    load_candidate_aligned_item_metadata,
    padding_and_clip,
)


class KRMBSeqReader(BaseReader):
    """
    KuaiRand Multi-Behavior Data Reader
    """

    @staticmethod
    def parse_data_args(parser):
        """
        args:
        - user_meta_file
        - item_meta_file
        - max_hist_seq_len
        - val_holdout_per_user
        - test_holdout_per_user
        - meta_file_sep
        - from BaseReader:
            - train_file
            - val_file
            - test_file
            - n_worker
        """
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--user_meta_file', type=str, required=True,
                            help='user raw feature file_path')
        parser.add_argument('--item_meta_file', type=str, required=True,
                            help='item raw feature file_path')
        parser.add_argument('--max_hist_seq_len', type=int, default=100,
                            help='maximum history length in the sample')
        parser.add_argument('--val_holdout_per_user', type=int, default=5,
                            help='number of holdout records for val set')
        parser.add_argument('--test_holdout_per_user', type=int, default=5,
                            help='number of holdout records for test set')
        parser.add_argument('--meta_file_sep', type=str, default=',',
                            help='separater of user/item meta csv file')
        parser.add_argument('--environment_split', type=str, default='all',
                            choices=['all', 'train', 'test'],
                            help='which disjoint half of the interaction log is used to fit the simulator')
        parser.add_argument('--train_environment_ratio', type=float, default=0.5,
                            help='fraction of interactions assigned to the train-environment half')
        parser.add_argument('--environment_split_seed', type=int, default=619607,
                            help='seed used only to balance per-user split-boundary records')
        parser.add_argument('--dataset_dir', type=str, default='',
                            help='directory containing item_types.csv and user_pop_ratio.csv; '
                                 'defaults to the directory of train_file')
        return parser

    def log(self):
        super().log()
        print(f"\tval_holdout_per_user: {self.val_holdout_per_user}")
        print(f"\ttest_holdout_per_user: {self.test_holdout_per_user}")

    def __init__(self, args):
        """
        - max_hist_seq_len
        - val_holdout_per_user
        - test_holdout_per_user
        - from BaseReader:
            - phase
            - n_worker
        """
        print("initiate KuaiRandMultiBehaior sequence reader")
        self.max_hist_seq_len = args.max_hist_seq_len
        self.val_holdout_per_user = args.val_holdout_per_user
        self.test_holdout_per_user = args.test_holdout_per_user
        self.environment_split = args.environment_split
        self.train_environment_ratio = args.train_environment_ratio
        self.environment_split_seed = args.environment_split_seed
        if self.environment_split != 'all' and not 0. < self.train_environment_ratio < 1.:
            raise ValueError("train_environment_ratio must be in (0, 1) when environment_split is train/test")

        # Load user popularity preferences and item types.
        dataset_dir = args.dataset_dir or os.path.dirname(os.path.abspath(args.train_file))
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.user_pop_ratios = pd.read_csv(
            os.path.join(self.dataset_dir, 'user_pop_ratio.csv')
        ).to_numpy().reshape(-1)
        super().__init__(args)
        aligned_item_metadata = load_candidate_aligned_item_metadata(
            self.dataset_dir, self
        )
        self.item_types = aligned_item_metadata['item_types']
        self.item_popularity = aligned_item_metadata['item_popularity']
        self.item_metadata_alignment = {
            key: aligned_item_metadata[key] for key in (
                'alignment_mode', 'max_popularity_error',
                'naive_type_agreement',
            )
        }
        print('Item metadata alignment:', self.item_metadata_alignment)

    def _read_data(self, args):
        """
        - log_data: pd.DataFrame
        - data: {'train': [row_id], 'val': [row_id], 'test': [row_id]}
        - users: [user_id]
        - user_id_vocab: {user_id: encoded_user_id}
        - user_meta: {user_id: {feature_name: feature_value}}
        - user_vocab: {feature_name: {feature_value: one-hot vector}}
        - selected_user_features
        - items: [item_id]
        - item_id_vocab: {item_id: encoded_item_id}
        - item_meta: {item_id: {feature_name: feature_value}}
        - item_vocab: {feature_name: {feature_value: one-hot vector}}
        - selected_item_features: [feature_name]
        - padding_item_meta: {feature_name: 0}
        - user_history: {uid: [row_id]}
        - response_list: [response_type]
        - padding_response: {response_type: 0}
        -
        """
        # read data_file
        print(f"Loading data files")
        self.log_data = pd.read_table(args.train_file, sep=args.data_separator)

        print("Load item meta data")
        item_meta_file = pd.read_csv(args.item_meta_file, sep=args.meta_file_sep)

        # Map raw item IDs to their corresponding data rows.
        self.item_meta = item_meta_file.set_index('video_id').to_dict('index')
        print("Load user meta data")
        user_meta_file = pd.read_csv(args.user_meta_file, sep=args.meta_file_sep)
        self.user_meta = user_meta_file.set_index('user_id').to_dict('index')

        # Build vocabularies from the complete log so that train/test simulator
        # checkpoints expose identical user/item ID spaces. Only target rows and
        # their histories are partitioned below.
        self.users = list(self.log_data['user_id'].unique())
        self.items = list(self.log_data['video_id'].unique())

        # The source file is already chronological within user. Sort explicitly
        # when a timestamp is available so the split remains causal if file row
        # order changes in a future preprocessing run.
        if 'time_ms' in self.log_data.columns:
            ordered_log = self.log_data.sort_values(['user_id', 'time_ms'], kind='stable')
        else:
            ordered_log = self.log_data
        grouped_rows = ordered_log.groupby('user_id', sort=False).groups
        full_user_history = {uid: list(grouped_rows[uid]) for uid in self.users}
        self.environment_indices, self.user_history = self._split_environment_data(full_user_history)

        # id reindex
        self.user_id_vocab = {uid: i + 1 for i, uid in enumerate(self.users)}
        self.item_id_vocab = {iid: i + 1 for i, iid in enumerate(self.items)}

        # selected meta features
        self.selected_item_features = ['video_type', 'music_type', 'upload_type', 'tag']
        self.selected_user_features = ['user_active_degree', 'is_live_streamer', 'is_video_author',
                                       'follow_user_num_range', 'fans_user_num_range',
                                       'friend_user_num_range', 'register_days_range'] \
                                      + [f'onehot_feat{fid}' for fid in [0, 1, 6, 9, 10, 11]]

        # meta feature vocabulary, {feature_name: {feature_value: one-hot/multi-hot vector}}
        # Vocabularies for one-hot encoding user and item features.
        self.user_vocab = get_onehot_vocab(user_meta_file, self.selected_user_features)
        self.item_vocab = get_onehot_vocab(item_meta_file, self.selected_item_features[:-1])
        self.item_vocab.update(get_multihot_vocab(item_meta_file, ['tag']))
        self.padding_item_meta = {f: np.zeros_like(list(v_dict.values())[0]) \
                                  for f, v_dict in self.item_vocab.items()}

        # response meta
        # Seven user-feedback signals.
        self.response_list = ['is_click', 'long_view', 'is_like', 'is_comment',
                              'is_forward', 'is_follow', 'is_hate']
        self.response_dim = len(self.response_list)
        self.padding_response = {resp: 0. for i, resp in enumerate(self.response_list)}
        self.response_neg_sample_rate = self.get_response_weights()

        # {'train': [row_id], 'val': [row_id], 'test': [row_id]}
        # Split the training, validation, and test sets.
        self.data = self._sequence_holdout(args)

    def _split_environment_data(self, full_user_history):
        """Create deterministic, disjoint, causal train/test simulator halves.

        Every user's earlier records are assigned to the train-environment half
        and later records to the test-environment half. A largest-remainder
        allocation makes the global row count match the requested ratio exactly;
        the seed only resolves equal fractional remainders and never shuffles a
        user's chronology.
        """
        if self.environment_split == 'all':
            indices = np.asarray(self.log_data.index, dtype=np.int64)
            return indices, full_user_history

        counts = np.asarray([len(full_user_history[u]) for u in self.users], dtype=np.int64)
        exact_counts = counts * self.train_environment_ratio
        train_counts = np.floor(exact_counts).astype(np.int64)
        target_train_size = int(round(len(self.log_data) * self.train_environment_ratio))
        rows_to_allocate = target_train_size - int(train_counts.sum())

        if rows_to_allocate > 0:
            rng = np.random.default_rng(self.environment_split_seed)
            tie_breakers = rng.random(len(self.users))
            fractional = exact_counts - train_counts
            allocation_order = np.lexsort((tie_breakers, -fractional))
            for user_pos in allocation_order[:rows_to_allocate]:
                if train_counts[user_pos] < counts[user_pos]:
                    train_counts[user_pos] += 1

        split_history = {}
        selected_rows = []
        for user_pos, uid in enumerate(self.users):
            rows = full_user_history[uid]
            boundary = int(train_counts[user_pos])
            if self.environment_split == 'train':
                selected = rows[:boundary]
            else:
                selected = rows[boundary:]
            split_history[uid] = selected
            selected_rows.extend(selected)

        selected_rows = np.asarray(sorted(selected_rows), dtype=np.int64)
        print(
            f"environment split: {self.environment_split} "
            f"({len(selected_rows)}/{len(self.log_data)} rows, "
            f"train_environment_ratio={self.train_environment_ratio:.4f}, "
            f"seed={self.environment_split_seed})"
        )
        return selected_rows, split_history

    # Split the training, validation, and test sets.
    def _sequence_holdout(self, args):
        """
        Holdout validation and test set from log_data
        """
        print(f"sequence holdout for users (-1, {args.val_holdout_per_user}, {args.test_holdout_per_user})")
        if args.val_holdout_per_user == 0 and args.test_holdout_per_user == 0:
            return {"train": self.environment_indices.copy(),
                    "val": np.asarray([], dtype=np.int64),
                    "test": np.asarray([], dtype=np.int64)}
        data = {"train": [], "val": [], "test": []}
        for u in tqdm(self.users):
            user_rows = self.user_history[u]
            n_train = len(user_rows) - args.val_holdout_per_user - args.test_holdout_per_user
            if n_train < 0.6 * len(user_rows):# or n_train < 10:
                continue
            val_end = n_train + args.val_holdout_per_user
            data['train'].append(user_rows[:n_train])
            data['val'].append(user_rows[n_train:val_end])
            data['test'].append(user_rows[val_end:])
        # Flatten each data split into a one-dimensional array.
        for k, v in data.items():
            data[k] = np.concatenate(v).astype(np.int64) if v else np.asarray([], dtype=np.int64)
        return data

    # Compute weights for each feedback type.
    def get_response_weights(self):
        ratio = {}
        environment_log = self.log_data.loc[self.environment_indices]
        for f in self.response_list:
            counts = environment_log[f].value_counts()
            # Divide the number of positive samples by the number of negatives.
            n_positive = int(counts.get(1, 0))
            n_negative = int(counts.get(0, 0))
            if n_positive == 0 or n_negative == 0:
                raise ValueError(
                    f"environment split '{self.environment_split}' has no positive or negative samples for {f}"
                )
            ratio[f] = float(n_positive) / n_negative
        ratio['is_hate'] *= -1
        return ratio

    ###########################
    #        Iterator         #
    ###########################

    def __getitem__(self, idx):
        """
        sample getter

        train batch after collate:
        {
            'user_id': (B,)
            'item_id': (B,)
            'is_click', 'long_view', ...: (B,)
            'uf_{feature}': (B,F_dim(feature)), user features
            'if_{feature}': (B,F_dim(feature)), item features
            'history': (B,max_H)
            'history_length': (B,)
            'history_if_{feature}': (B, max_H, F_dim(feature))
            'history_{response}': (B, max_H)
            'loss_weight': (B, n_response)
        }
        """
        row_id = self.data[self.phase][idx]
        row = self.log_data.iloc[row_id]

        user_id = row['user_id']  # raw user ID
        item_id = row['video_id']  # raw item ID

        # user, item, responses
        record = {
            'user_id': self.user_id_vocab[row['user_id']],  # encoded user ID
            'item_id': self.item_id_vocab[row['video_id']],  # encoded item ID
            # The auxiliary file is aligned once to the encoded-item vocabulary
            # during reader construction; never index its raw CSV row order here.
            'item_type': self.item_types[
                self.item_id_vocab[row['video_id']] - 1
            ],
        }

        # Compute the loss weight for each feedback type.
        for _, f in enumerate(self.response_list):
            record[f] = row[f]
        loss_weight = np.array([1. if record[f] == 1 else -self.response_neg_sample_rate[f]
        if f == 'is_hate' else self.response_neg_sample_rate[f] for i, f in enumerate(self.response_list)])
        record["loss_weight"] = loss_weight

        # meta features
        user_meta = self.get_user_meta_data(user_id)
        record.update(user_meta)
        item_meta = self.get_item_meta_data(item_id)
        record.update(item_meta)

        # history features (max_H,)
        H_rowIDs = [rid for rid in self.user_history[user_id] if rid < row_id][-self.max_hist_seq_len:]
        history, hist_length, hist_meta, hist_response, user_pop_prefer = self.get_user_history(H_rowIDs)
        record['history'] = np.array(history)
        record['history_length'] = hist_length
        record['user_pop_prefer'] = user_pop_prefer
        for f, v in hist_meta.items():
            record[f'history_{f}'] = v
        for f, v in hist_response.items():
            record[f'history_{f}'] = v

        return record

    # Return the one-hot encoding of user features.
    def get_user_meta_data(self, user_id):
        """
        @input:
        - user_id: raw user ID
        @output:
        - user_meta_record: {'uf_{feature_name}: one-hot vector'}
        """
        user_feature_dict = self.user_meta[user_id]
        user_meta_record = {f'uf_{f}': self.user_vocab[f][user_feature_dict[f]] \
                            for f in self.selected_user_features}
        return user_meta_record

    # Return the one-hot encoding of item features.
    def get_item_meta_data(self, item_id):
        """
        @input:
        - item_id: raw item ID
        @output:
        - item_meta_record: {'if_{feature_name}: one-hot vector'}
        """
        item_feature_dict = self.item_meta[item_id]
        item_meta_record = {f'if_{f}': self.item_vocab[f][item_feature_dict[f]] \
                            for f in self.selected_item_features[:-1]}
        item_meta_record['if_tag'] = np.sum([self.item_vocab['tag'][tag_id] \
                                             for tag_id in item_feature_dict['tag'].split(',')], axis=0)
        return item_meta_record

    def get_user_history(self, H_rowIDs):
        """
        @input:
        - H_rowIDs: [idx of log_data]
        @output:
        - history: [encoded item ID]
        - L: history length (less than or equals to max_hist_seq_len)
        - hist_meta: {if_{feature_name}: (max_hist_seq_len, feature_dim)}
        - history_response: {response_type: (max_hist_seq_len,)}
        """
        L = len(H_rowIDs)
        user_pop_prefer = 0.
        if L == 0:
            # (max_H,)
            history = [0] * self.max_hist_seq_len
            # {if_{feature_name}: (max_H, feature_dim)
            hist_meta = {f'if_{f}': np.tile(self.padding_item_meta[f], self.max_hist_seq_len) \
                         for f in self.selected_item_features}
            # {resp_type: (max_H)}
            history_response = {resp: np.array([self.padding_response[resp]] * self.max_hist_seq_len) \
                                for resp in self.response_list}
        else:
            H = self.log_data.iloc[H_rowIDs]
            # list of encoded iid
            item_ids = [self.item_id_vocab[iid] for iid in H['video_id']]
            # (max_H,)
            history = padding_and_clip(item_ids, self.max_hist_seq_len)
            # [{if_{feature}: one-hot vector}]
            meta_list = [self.get_item_meta_data(iid) for iid in H['video_id']]
            # history item meta features: {if_{feature_name}: }
            hist_meta = {}
            # Pad user histories shorter than max_hist_seq_len.
            for f in self.selected_item_features:
                padding = [self.padding_item_meta[f] for i in range(self.max_hist_seq_len - L)]
                real_hist = [v_dict[f'if_{f}'] for v_dict in meta_list]
                # {if_{feature_name}: (max_H, feature_dim)}
                hist_meta[f'if_{f}'] = np.concatenate(padding + real_hist, axis=0)
            # {resp_type: (max_H,)}
            history_response = {}
            for resp in self.response_list:
                padding = np.array([self.padding_response[resp]] * (self.max_hist_seq_len - L))
                real_resp = np.array(H[resp])
                history_response[resp] = np.concatenate([padding, real_resp], axis=0)
            # Compute the user's current popularity preference.
            # Count only items that the user clicked previously.

            H = H[H['is_click'] > 0]
            item_ids = [self.item_id_vocab[iid] for iid in H['video_id']]
            discount = math.pow(math.e, -1./2)
            for item_id in item_ids:
                user_pop_prefer = (user_pop_prefer + self.item_types[item_id - 1])
                user_pop_prefer = user_pop_prefer * discount
            L = len(item_ids)
            if L > 0:
                user_pop_prefer = user_pop_prefer * (1 - discount) / (discount - math.pow(discount, L + 1))
                #user_pop_prefer = user_pop_prefer / L
        return history, L, hist_meta, history_response, user_pop_prefer

    def get_item_catalog(self):
        """Return every item once in encoded-ID order for balanced calibration."""
        meta_records = [self.get_item_meta_data(raw_id) for raw_id in self.items]
        catalog = {
            'item_id': np.arange(1, len(self.items) + 1, dtype=np.int64),
            'item_type': np.asarray(self.item_types, dtype=np.int64),
        }
        for feature in self.selected_item_features:
            key = f'if_{feature}'
            catalog[key] = np.stack(
                [record[key] for record in meta_records]
            ).astype(np.float32, copy=False)
        return catalog

    # Return statistics for the data in use.
    def get_statistics(self):
        """
        - n_user
        - n_item
        - s_parsity
        - from BaseReader:
            - length
            - fields
        """
        stats = {}
        stats["raw_data_size"] = len(self.environment_indices)
        stats["full_data_size"] = len(self.log_data)
        stats["environment_split"] = self.environment_split
        stats["train_environment_ratio"] = self.train_environment_ratio
        stats["data_size"] = [len(self.data['train']), len(self.data['val']), len(self.data['test'])]
        stats["n_user"] = len(self.users)
        stats["n_item"] = len(self.items)
        # Maximum history length used to encode the user state.
        stats["max_seq_len"] = self.max_hist_seq_len
        stats["user_features"] = self.selected_user_features
        stats["user_feature_dims"] = {f: len(list(v_dict.values())[0]) for f, v_dict in self.user_vocab.items()}
        stats["item_features"] = self.selected_item_features
        stats["item_feature_dims"] = {f: len(list(v_dict.values())[0]) for f, v_dict in self.item_vocab.items()}
        stats["feedback_type"] = self.response_list
        stats["feedback_size"] = self.response_dim
        stats["feedback_negative_sample_rate"] = self.response_neg_sample_rate
        stats["item_metadata_alignment"] = self.item_metadata_alignment
        return stats
