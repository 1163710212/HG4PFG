import numpy as np
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader, Subset
from torch.distributions import Categorical
import torch.nn.functional as F

import utils
from reader import *
from model.simulator import *
from env.BaseRLEnvironment import BaseRLEnvironment
import pandas as pd
import math
import os


class KREnvironment_WholeSession_GPU(BaseRLEnvironment):
    '''
    KuaiRand simulated environment for consecutive list-wise recommendation
    Main interface:
    - parse_model_args: for hyperparameters
    - reset: reset online environment, monitor, and corresponding initial observation
    - step: action --> new observation, user feedbacks, and other updated information
    - get_candidate_info: obtain the entire item candidate pool
    Main Components:
    - data reader: self.reader for user profile&history sampler
    - user immediate response model: see self.get_response
    - no user leave model: see self.get_leave_signal
    - candidate item pool: self.candidate_ids, self.candidate_item_meta
    - history monitor: self.env_history, not set up until self.reset
    '''

    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - uirm_log_path
        - slate_size
        - episode_batch_size
        - item_correlation
        - ad_temper_penalty
        - single_response
        - from BaseRLEnvironment
            - max_step_per_episode
            - initial_temper
        """
        parser = BaseRLEnvironment.parse_model_args(parser)
        parser.add_argument('--uirm_log_path', type=str, required=True,
                            help='log path for pretrained user immediate response model')
        parser.add_argument('--slate_size', type=int, required=6,
                            help='number of item per recommendation slate')
        parser.add_argument('--episode_batch_size', type=int, default=32,
                            help='episode sample batch size')
        parser.add_argument('--item_correlation', type=float, default=0,
                            help='magnitude of item correlation')
        parser.add_argument(
            '--ad_temper_penalty',
            type=float,
            default=1.0,
            help='amount subtracted from user temper when AD reaches ad_bound',
        )
        parser.add_argument('--single_response', action='store_true',
                            help='only include the first feedback as reward signal')
        parser.add_argument('--dataset_dir', type=str, default='',
                            help='directory containing item_types.csv and user_pop_ratio.csv')
        parser.add_argument(
            '--align_item_metadata', action='store_true',
            help='validate and align popularity rows to the encoded item order',
        )
        return parser

    def __init__(self, args):
        """
        from BaseRLEnvironment:
            self.max_step_per_episode
            self.initial_temper
        self.uirm_log_path
        self.slate_size
        self.rho
        self.immediate_response_stats: reader statistics for user response model
        self.immediate_response_model: the ground truth user response model
        self.max_hist_len
        self.response_types
        self.response_dim: number of feedback_type
        self.response_weights
        self.reader
        self.candidate_iids: [encoded item id]
        self.candidate_item_meta: {'if_{feature_name}': (n_item, feature_dim)}
        self.n_candidate
        self.candidate_item_encoding: (n_item, item_enc_dim)
        self.gt_state_dim: ground truth user state vector dimension
        self.action_dim: slate size
        self.observation_space: see reader.get_statistics()
        self.action_space: n_condidate
        """
        super(KREnvironment_WholeSession_GPU, self).__init__(args)
        self.uirm_log_path = args.uirm_log_path
        self.slate_size = args.slate_size
        self.episode_batch_size = args.episode_batch_size
        self.rho = args.item_correlation
        self.single_response = args.single_response
        # 公平性退出机制，限制边界
        self.ad_bound = args.ad_bound
        self.ad_temper_penalty = args.ad_temper_penalty
        if (
            not math.isfinite(self.ad_temper_penalty)
            or self.ad_temper_penalty < 0.0
        ):
            raise ValueError("ad_temper_penalty must be finite and non-negative")

        infile = open(args.uirm_log_path, 'r')
        class_args = eval(infile.readline())  # example: Namespace(model='RL4RSUserResponse', reader='RL4RSDataReader')
        model_args = eval(infile.readline())  # model parameters in Namespace
        print("Environment arguments: \n" + str(model_args))
        infile.close()
        print("Loading raw data")
        assert (class_args.reader == 'KRMBSeqReader' 
                or class_args.reader == 'MLSeqReader'
                or class_args.reader == 'RecKRMBSeqReader'
                ) and 'KRMBUserResponse' in class_args.model

        # Read auxiliary data from the same dataset directory recorded by the
        # user-response checkpoint. This keeps train/test environment contracts
        # portable and prevents accidental cross-dataset loading.
        checkpoint_dataset_dir = getattr(model_args, 'dataset_dir', '')
        dataset_dir = args.dataset_dir or checkpoint_dataset_dir
        if not dataset_dir:
            raise ValueError("dataset_dir is required by KREnvironment_WholeSession_GPU")
        dataset_dir = os.path.realpath(dataset_dir)
        checkpoint_dataset_dir = os.path.realpath(checkpoint_dataset_dir) if checkpoint_dataset_dir else ''
        if checkpoint_dataset_dir and checkpoint_dataset_dir != dataset_dir:
            raise ValueError(
                f"dataset_dir mismatch: RL args use {dataset_dir}, "
                f"but the response-model checkpoint uses {checkpoint_dataset_dir}"
            )
        args.dataset_dir = dataset_dir
        self.dataset_dir = dataset_dir
        self.user_pop_ratios = torch.tensor(
            pd.read_csv(os.path.join(dataset_dir, 'user_pop_ratio.csv')).to_numpy()
        ).to(self.device)
        self.item_types = torch.tensor(
            pd.read_csv(os.path.join(dataset_dir, 'item_types.csv')).to_numpy()
        ).to(self.device)

        print("Load user sequence reader")
        reader, reader_args = self.get_reader(args.uirm_log_path, args.dataset)  # definition in base
        # 数据读取器
        self.reader = reader
        self.item_metadata_alignment = {
            'alignment_mode': 'legacy_file_order',
            'max_popularity_error': None,
            'naive_type_agreement': 1.0,
        }
        if getattr(args, 'align_item_metadata', False):
            aligned = utils.load_candidate_aligned_item_metadata(
                dataset_dir, reader
            )
            self.item_types = torch.as_tensor(
                aligned['item_types'], dtype=torch.long, device=self.device
            ).reshape(-1, 1)
            self.item_popularity = torch.as_tensor(
                aligned['item_popularity'], dtype=torch.float32,
                device=self.device,
            )
            self.item_popularity_percentile = torch.as_tensor(
                aligned['popularity_percentile'], dtype=torch.float32,
                device=self.device,
            )
            reader.item_types = aligned['item_types'].copy()
            self.item_metadata_alignment = {
                key: aligned[key] for key in (
                    'alignment_mode', 'max_popularity_error',
                    'naive_type_agreement',
                )
            }
            print('Item metadata alignment:', self.item_metadata_alignment)
        else:
            self.item_popularity = torch.as_tensor(
                pd.read_csv(
                    os.path.join(dataset_dir, 'item_popularity.csv')
                )['popularity'].to_numpy(),
                dtype=torch.float32, device=self.device,
            )
            self.item_popularity_percentile = torch.as_tensor(
                pd.Series(self.item_popularity.detach().cpu().numpy()).rank(
                    method='average', pct=True
                ).to_numpy(),
                dtype=torch.float32, device=self.device,
            )
        print(self.reader.get_statistics())

        print("Load immediate user response model")
        uirm_stats, uirm_model, uirm_args = self.get_user_model(args.uirm_log_path, args.device)  # definition in base
        self.immediate_response_stats = uirm_stats
        self.immediate_response_model = uirm_model
        self.max_hist_len = uirm_stats['max_seq_len']
        self.response_types = uirm_stats['feedback_type']
        self.response_dim = len(self.response_types)
        self.response_weights = torch.tensor(list(self.reader.get_response_weights().values())).to(torch.float).to(
            args.device)
        if args.single_response:
            self.response_weights = torch.zeros_like(self.response_weights)
            self.response_weights[0] = 1

        print("Setup candidate item pool")

        # [encoded item id], size (n_item,), [1, 2, ..., num_item + 1]
        self.candidate_iids = torch.tensor([reader.item_id_vocab[iid] for iid in reader.items]).to(self.device)

        # item meta: {'if_{feature_name}': (n_item, feature_dim)}
        # 返回物品集特征的one-hot编码字典
        # {item_1:{feature_1:[], feature_2:[]...}, item_2:{feature_1:[], feature_2:[]...}, ...}
        candidate_meta = [reader.get_item_meta_data(iid) for iid in reader.items]
        self.candidate_item_meta = {}
        self.n_candidate = len(candidate_meta)
        # [n_item, feature_dim]
        for k in candidate_meta[0]:
            self.candidate_item_meta[k] = torch.FloatTensor(np.concatenate([meta[k] for meta in candidate_meta])) \
                .view(self.n_candidate, -1).to(self.device)

        # (n_item, item_enc_dim), groud truth encoding is implicit to RL agent
        # 获取整个物品集的嵌入编码矩阵,[1, n_item, item_enc_dim]
        item_enc, _ = self.immediate_response_model.get_item_encoding(self.candidate_iids,
                                                                      {k[3:]: v for k, v in
                                                                       self.candidate_item_meta.items()}, 1)
        # [n_item, item_enc_dim]
        self.candidate_item_encoding = item_enc.view(-1, self.immediate_response_model.enc_dim)

        # spaces
        self.gt_state_dim = self.immediate_response_model.state_dim
        # 动作空间维度为推荐列表长度
        self.action_dim = self.slate_size
        self.observation_space = self.reader.get_statistics()
        self.action_space = self.n_candidate

        self.immediate_response_model.to(args.device)
        self.immediate_response_model.device = args.device

    def get_candidate_info(self, feed_dict, all_item=True):
        """
        Add entire item pool as candidate for the feed_dict
        @input:
        - all_item: whether obtain all item features from candidate pool
        - feed_dict
        @output:
        - candidate_info: {'item_id': (L,),
                           'if_{feature_name}': (n_item, feature_dim)}
        """
        if all_item:
            candidate_info = {'item_id': self.candidate_iids}
            candidate_info.update(self.candidate_item_meta)
        else:
            candidate_info = {'item_id': feed_dict['item_id']}
            indices = feed_dict['item_id'] - 1
            candidate_info.update({k: v[indices] for k, v in self.candidate_item_meta.items()})
        return candidate_info

    def reset(self, params=None):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar, 
                   'empty_history': True if start from empty history, 
                   'initial_history': start with initial history}
        @process:
        - self.batch_iter
        - self.current_observation
        - self.current_step
        - self.current_temper
        - self.env_history
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        '''
        params = {} if params is None else dict(params)
        if 'empty_history' not in params:
            params['empty_history'] = False

        # set inference batch size
        if 'batch_size' in params:
            BS = params['batch_size']
        else:
            BS = self.episode_batch_size

        sample_indices = params.get('sample_indices')
        if sample_indices is not None:
            sample_indices = list(sample_indices)
            if len(sample_indices) != BS:
                raise ValueError(
                    "sample_indices must contain exactly batch_size entries"
                )
            if BS != self.episode_batch_size:
                raise ValueError(
                    "A fixed-user reset requires batch_size to equal "
                    "episode_batch_size"
                )
            sample_dataset = Subset(self.reader, sample_indices)
            self.batch_iter = iter(DataLoader(
                sample_dataset,
                batch_size=BS,
                shuffle=False,
                pin_memory=True,
                num_workers=4,
            ))
        else:
            # Standard online-training reset: sample interaction rows freely.
            self.batch_iter = iter(DataLoader(
                self.reader,
                batch_size=BS,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
            ))
        sample_info = next(self.batch_iter)
        # 从历史数据中获取初始用户
        # {'user_profile': {'user_id': (B,), 'uf_{feature_name}': (B, feature_dim)},
        # 'user_history':  {'history': (B, max_H),
        #                'history_if_{feature_name}': (B, max_H, feature_dim),
        #                'history_{response}': (B, max_H),
        #                'history_length': (B, )}
        self.sample_batch = self.get_observation_from_batch(sample_info)
        self.current_observation = self.sample_batch
        self.current_step = torch.zeros(self.episode_batch_size).to(self.device)
        self.cum_cost = torch.zeros(self.episode_batch_size).to(self.device)
        self.current_sample_head_in_batch = BS

        # user temper for leave model
        self.current_temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
        self.current_pop_temper = torch.ones(self.episode_batch_size).to(self.device) * 5
        self.current_sum_reward = torch.zeros(self.episode_batch_size).to(self.device)

        # batch-wise monitor, 添加用户流行度偏好、上层智能体权重
        self.env_history = {'step': [0.], 'leave': [], 'temper': [],
                            'coverage': [], 'ILD': [], 'pop_ratio': [], 'ad': []}

        return deepcopy(self.current_observation)

    def reset_for_unique_users(self, n_users, seed):
        """Reset with one sampled interaction row for each distinct user.

        The ordinary environment reset samples interaction rows and can therefore
        include the same user more than once. Final user-level evaluation needs
        exactly ``n_users`` distinct initial users, so it first samples user IDs
        without replacement and then samples one eligible row for each user.
        """
        if n_users <= 0:
            raise ValueError("n_users must be positive")
        if n_users != self.episode_batch_size:
            raise ValueError(
                "n_users must equal the test environment episode_batch_size"
            )

        phase_row_ids = np.asarray(
            self.reader.data[self.reader.phase], dtype=np.int64
        )
        if phase_row_ids.size == 0:
            raise ValueError("The test reader contains no eligible interaction rows")

        raw_user_ids = self.reader.log_data.iloc[phase_row_ids][
            'user_id'
        ].to_numpy()
        positions_by_user = {}
        for sample_position, raw_user_id in enumerate(raw_user_ids):
            positions_by_user.setdefault(raw_user_id, []).append(sample_position)

        unique_user_ids = list(positions_by_user)
        if len(unique_user_ids) < n_users:
            raise ValueError(
                f"Requested {n_users} unique test users, but only "
                f"{len(unique_user_ids)} are available"
            )

        rng = np.random.default_rng(seed)
        selected_user_positions = rng.choice(
            len(unique_user_ids), size=n_users, replace=False
        )
        sample_indices = []
        for user_position in selected_user_positions:
            raw_user_id = unique_user_ids[int(user_position)]
            eligible_positions = positions_by_user[raw_user_id]
            row_position = int(rng.integers(len(eligible_positions)))
            sample_indices.append(eligible_positions[row_position])

        observation = self.reset({
            'batch_size': n_users,
            'sample_indices': sample_indices,
        })
        encoded_user_ids = observation['user_profile']['user_id']
        if torch.unique(encoded_user_ids).numel() != n_users:
            raise RuntimeError("Unique-user test sampling produced duplicate users")
        return observation

    def reset_for_popular_users(self, n_users, seed, min_popular_preference):
        """Reset with distinct users selected only by their initial history.

        One eligible row is sampled uniformly for each shuffled user before
        applying the popularity threshold. Selection therefore cannot depend
        on any recommendation outcome produced by the evaluated policy.
        """
        if n_users <= 0:
            raise ValueError("n_users must be positive")
        if n_users != self.episode_batch_size:
            raise ValueError(
                "n_users must equal the test environment episode_batch_size"
            )
        if not 0.0 <= min_popular_preference <= 1.0:
            raise ValueError("min_popular_preference must be in [0, 1]")

        phase_row_ids = np.asarray(
            self.reader.data[self.reader.phase], dtype=np.int64
        )
        raw_user_ids = self.reader.log_data.iloc[phase_row_ids][
            'user_id'
        ].to_numpy()
        positions_by_user = {}
        for sample_position, raw_user_id in enumerate(raw_user_ids):
            positions_by_user.setdefault(raw_user_id, []).append(sample_position)

        rng = np.random.default_rng(seed)
        user_order = rng.permutation(list(positions_by_user))
        sample_indices = []
        for raw_user_id in user_order:
            eligible_positions = positions_by_user[raw_user_id]
            sample_position = int(rng.choice(eligible_positions))
            initial_record = self.reader[sample_position]
            if float(initial_record['user_pop_prefer']) < min_popular_preference:
                continue
            sample_indices.append(sample_position)
            if len(sample_indices) == n_users:
                break

        if len(sample_indices) < n_users:
            raise ValueError(
                f"Only {len(sample_indices)} distinct users met initial "
                f"popular preference >= {min_popular_preference:.3f}"
            )
        observation = self.reset({
            'batch_size': n_users,
            'sample_indices': sample_indices,
        })
        initial_preference = observation['user_history']['user_pop_prefer']
        if torch.any(initial_preference < min_popular_preference - 1e-7):
            raise RuntimeError("Popular-cohort reset violated its threshold")
        if torch.unique(
            observation['user_profile']['user_id']
        ).numel() != n_users:
            raise RuntimeError("Popular-cohort reset produced duplicate users")
        return observation

    def step(self, step_dict):
        """
        users react to the recommendation action
        @input:
        - step_dict: {'action': (B, slate_size),
                      'action_features': (B, slate_size, item_dim) }
        @output:
        - new_observation: {'user_profile': {'user_id': (B,),
                                             'uf_{feature_name}': (B, feature_dim)},
                            'user_history': {'history': (B, max_H),
                                             'history_if_{feature_name}': (B, max_H, feature_dim),
                                             'history_{response}': (B, max_H),
                                             'history_length': (B, )}}
        - response_dict: {'immediate_response': (B, slate_size, n_feedback),
                          'user_state': (B, gt_state_dim),
                          'coverage': scalar,
                          'ILD': scalar,
                          'done': (B,)}
        - update_info: see self.update_observation@output - update_info
        """

        # URM forward
        with torch.no_grad():
            action = step_dict['action']  # must be indices on candidate_ids

            # get user response
            # 1.根据推荐模型产生的物品列表，返回用户反馈
            response_dict = self.get_response(step_dict)
            response = response_dict['immediate_response']

            # done mask and temper update
            # (B,)
            # 2.根据用户反馈，判断其是否会离开
            done_mask = self.get_leave_signal(None, action, response_dict)  # this will also change self.current_temper
            response_dict['done'] = done_mask
            response_dict['cum_cost'] = self.cum_cost

            # 3.update user history in current_observation
            # {'slate': (B, slate_size), 'updated_observation': a copy of self.current_observation}
            update_info = self.update_observation(None, action, response, done_mask)
            response_dict['cost'] = self.current_cost

            # 4.env_history update: step, leave, temper, converage, ILD
            self.current_step += 1
            n_leave = done_mask.sum()
            self.env_history['leave'].append(n_leave.item())
            self.env_history['temper'].append(torch.mean(self.current_temper).item())
            self.env_history['coverage'].append(response_dict['coverage'])
            # ILD: estimates the dissimilarity between items in each recommended list, based on item embedding.
            self.env_history['ILD'].append(response_dict['ILD'])

            # 5.when users left, new users come into the running batch
            if n_leave > 0:
                final_steps = self.current_step[done_mask].detach().cpu().numpy()
                for fst in final_steps:
                    self.env_history['step'].append(fst)

                if self.current_sample_head_in_batch + n_leave < self.episode_batch_size:
                    # reuse previous batch if there are sufficient samples for n_leave
                    # 当前所采样的用户，剩余个数大于n_leave，使用下n_leave个用户
                    head = self.current_sample_head_in_batch
                    tail = self.current_sample_head_in_batch + n_leave
                    # 将新用户的初始信息更新进来
                    for obs_key in ['user_profile', 'user_history']:
                        for k, v in self.sample_batch[obs_key].items():
                            self.current_observation[obs_key][k][done_mask] = v[head:tail]
                    self.current_sample_head_in_batch += n_leave
                else:
                    # sample new users to fill in the blank
                    sample_info = self.sample_new_batch_from_reader()
                    self.sample_batch = self.get_observation_from_batch(sample_info)
                    # 将新用户的初始信息更新进来
                    for obs_key in ['user_profile', 'user_history']:
                        for k, v in self.sample_batch[obs_key].items():
                            self.current_observation[obs_key][k][done_mask] = v[:n_leave]
                    self.current_sample_head_in_batch = n_leave
                self.current_step[done_mask] *= 0
                self.cum_cost[done_mask] *= 0
                self.current_temper[done_mask] *= 0
                self.current_temper[done_mask] += self.initial_temper
                self.current_pop_temper[done_mask] *= 0
                self.current_pop_temper[done_mask] += self.initial_temper
            else:
                self.env_history['step'].append(self.env_history['step'][-1])

        return deepcopy(self.current_observation), response_dict, update_info, self.current_step.clone()

    def get_response(self, step_dict):
        """
        @input:
        - step_dict: {'action': (B, slate_size)}
        @output:
        - response_dict: {'immediate_response': (B, slate_size, n_feedback),
                          'user_state': (B, gt_state_dim),
                          'coverage': scalar,
                          'ILD': scalar}
        """
        # actions (exposures), (B, slate_size), indices of self.candidate_iid
        action = step_dict['action']
        # 计算一个batch_size的物品覆盖度
        coverage = len(torch.unique(action))
        B = self.episode_batch_size

        ########################################
        # This is where the action take effect #
        # (B, action_dim, 1, enc_dim)
        batch = {'item_id': self.candidate_iids[action]}
        batch.update(self.current_observation['user_profile'])
        batch.update(self.current_observation['user_history'])
        batch.update({k: v[action] for k, v in self.candidate_item_meta.items()})
        # 预测用户对推荐列表中各个物品的反馈
        out_dict = self.immediate_response_model(batch)
        ########################################

        # (B, slate_size, n_feedback)
        behavior_scores = out_dict['probs']

        # (B, slate_size, item_dim)
        item_enc = self.candidate_item_encoding[action].view(B, self.slate_size, -1)
        item_enc_norm = F.normalize(item_enc, p=2.0, dim=-1)
        # (B, slate_size)， 计算推荐列表内部物品间的相似度
        corr_factor = self.get_intra_slate_similarity(item_enc_norm)

        # user response sampling
        # (B, slate_size, n_feedback). behavior_scores已经做了概率化处理，这里为啥还要通过一个sigmoid？
        # 由于训练时输出经过了两层sigmoid?
        #point_scores = torch.sigmoid(behavior_scores) - corr_factor.view(B, self.slate_size, 1) * self.rho
        point_scores = behavior_scores - corr_factor.view(B, self.slate_size, 1) * self.rho
        point_scores[point_scores < 0] = 0

        # (B, slate_size, n_feedback). torch.bernoulli一个离散分布，有两个结果，即成功和失败，各个维度返回1/0
        response = torch.bernoulli(point_scores).detach()

        return {'immediate_response': response,
                'user_state': out_dict['state'],
                # describes the number of distinct items exposed in a mini-batch.
                'coverage': coverage,
                # estimates the dissimilarity between items in each recommended list, based on item embedding.
                'ILD': 1 - torch.mean(corr_factor).item()}

    # 在跨session推荐环境会用到
    def get_ground_truth_user_state(self, profile, history):
        batch_data = {}
        batch_data.update(profile)
        batch_data.update(history)
        gt_state_dict = self.immediate_response_model.encode_state(batch_data, self.episode_batch_size)
        gt_user_state = gt_state_dict['state'].view(self.episode_batch_size, 1, self.gt_state_dim)
        return gt_user_state

    # 计算推荐列表内部物品间的相似度
    def get_intra_slate_similarity(self, action_item_encoding):
        """
        @input:
        - action_item_encoding: (B, slate_size, enc_dim)
        @output:
        - similarity: (B, slate_size)
        """
        B, L, d = action_item_encoding.shape
        # pairwise similarity in a slate (B, L, L)
        pair_similarity = torch.mean(action_item_encoding.view(B, L, 1, d) * action_item_encoding.view(B, 1, L, d),
                                     dim=-1)
        # similarity to slate average, (B, L)
        point_similarity = torch.mean(pair_similarity, dim=-1)
        return point_similarity

    # 根据用户反馈，根据用户temper，temper小于一，用户会退出
    def get_leave_signal(self, user_state, action, response_dict):
        """
        User leave model maintains the user temper, and a user leaves when the temper drops below 1.
        @input:
        - user_state: not used in this env
        - action: not used in this env
        - response_dict: (B, slate_size, n_feedback)
        @process:
        - update temper
        @output:
        - done_mask: 
        """
        # (B, slate_size, n_feedback)
        point_reward = response_dict['immediate_response'] * self.response_weights.view(1, 1, -1)
        # (B, slate_size)
        combined_reward = torch.sum(point_reward, dim=2)
        # (B, )
        temper_boost = torch.mean(combined_reward, dim=1)

        # # 获取用户id和推荐物品id
        # user_id = self.current_observation['user_profile']['user_id'].reshape(-1)
        # item_id = action.reshape(-1)
        # item_type = self.item_types[item_id].reshape(action.shape[0], -1)

        # 获取当前用户流行度偏好、推荐物品类型
        user_pop_prefer = self.current_observation['user_history']['user_pop_prefer']#.reshape(-1, 1)
        item_id = action.reshape(-1)
        item_type = self.item_types[item_id].reshape(action.shape[0], -1).float()

        # 方式一：如果推荐列表中长尾物品比例和用户长尾偏好一致，增加更多的容忍度
        # index = (torch.abs(torch.mean((item_type + 0.), dim=1) - user_pop_prefer) >= 0.3)
        # self.current_temper[index] -= 1

        # 方式二：如果流行物品与非流行物品曝光比绝对误差大于0.3，用户退出  
        # pop_ratio = torch.mean(item_type, dim=1).cpu().numpy()
        # AD = torch.abs(pop_ratio - (1 - pop_ratio))
        # if AD >= 0.4:
        #     self.current_temper = 0
        
        #方式三：如果流行物品与非流行物品曝光比绝对误差大于阈值，用户容忍度下降  
        pop_ratio = torch.mean(item_type, dim=1)
        AD = torch.abs(pop_ratio - (1 - pop_ratio))
        #print(f'mush ad {torch.sum(AD >= 0.3)}')
        self.current_temper[AD >= self.ad_bound] -= self.ad_temper_penalty
        
        # 更新交互数据
        pop_ratio = torch.mean(item_type).cpu().numpy()
        AD = abs(pop_ratio - (1 - pop_ratio))
        self.env_history['pop_ratio'].append(pop_ratio)
        self.env_history['ad'].append(AD)
        # temper update for leave model
        # 混合奖励大于等于2，不降低用户temper；小于等于0，只降低2；0-2降低mean_combined_reward-2
        temper_update = temper_boost - 2
        temper_update[temper_update > 0] = 0
        temper_update[temper_update < -2] = -2
        self.current_temper += temper_update
        # leave signal
        done_mask = self.current_temper < 1
        return done_mask

    def update_observation(self, user_state, action, response, done_mask, update_current=True):
        """
        user profile stays static, only update user history
        @input:
        - user_state: not used in this env
        - action: (B, slate_size), indices of self.candidate_iids
        - response: (B, slate_size, n_feedback)
        - done_mask: not used in this env
        @output:
        - update_info: {slate: (B, slate_size),
                        updated_observation: same format as self.reset@output - observation}
        """
        # (B, slate_size), convert to encoded item id
        rec_list = self.candidate_iids[action]

        # history update，更新历史物品id和历史长度
        old_history = self.current_observation['user_history']
        
        # 更新用户流行度偏好
        item_type = ((self.item_types[action]).float().squeeze() * response[:, :, 0])#.mean(dim=1)
        discount = math.pow(math.e, -1./16)
        user_pop_prefer = old_history['user_pop_prefer']
        # 列表中用户有正反馈的物品数
        is_pos_click_num = (item_type.sum(dim=1).long() >= 1).int()#(item_type.sum(dim=1).long() >= 1).int()
        self.cum_cost += (torch.mean(item_type.float(), dim=1))
        self.current_cost = (torch.mean(item_type.float(), dim=1))
        L = old_history['history_length']
        #print(f'mush{item_type.mean(dim=1), torch.pow(discount, is_pos_click_num), L, ( 1 - torch.pow(discount, L - is_pos_click_num + 1e-5)) / (1 - torch.pow(discount, L + 1e-5))}')
        # for i in range(item_type.shape[1]):
        #     deta = (1 - discount) / (discount - torch.pow(discount, L + 1e-5  + 1)) * item_type[:, i]
        #     user_pop_prefer = (user_pop_prefer + deta) * torch.pow(discount, item_type[:, i])
        # s为常数值时使用
        user_pop_prefer = (user_pop_prefer + item_type.mean(dim=1) * (1 - discount) / (discount - torch.pow(discount, L + 1e-5  + 1))) * torch.pow(discount, is_pos_click_num)
        user_pop_prefer = user_pop_prefer * (1 - torch.pow(discount, L + 1e-5)) / (1 - torch.pow(discount, L + is_pos_click_num + 1e-5))
        #s为inf时使用
        # user_pop_prefer = (user_pop_prefer * L + item_type.mean(dim=1)) / (L + 1e-5 + is_pos_click_num)

        # 更点击历史长度
        max_H = self.max_hist_len
        L += is_pos_click_num
        #L[L > max_H] = max_H

        new_history = {'history': torch.cat((old_history['history'], rec_list), dim=1)[:, -max_H:],
                       'history_length': L, 'user_pop_prefer': user_pop_prefer}

        # history item features，更新历史物品特征
        for k, candidate_meta_features in self.candidate_item_meta.items():
            # (B, slate_size, feature_dim)
            meta_features = candidate_meta_features[action]
            # (B, max_H, feature_dim)
            previous_meta = old_history[f'history_{k}'].view(self.episode_batch_size, max_H, -1)
            new_history[f'history_{k}'] = torch.cat((previous_meta, meta_features), dim=1)[:, -max_H:, :].view(
                self.episode_batch_size, -1)

        # history item responses，更新历史反馈
        for i, R in enumerate(self.immediate_response_model.feedback_types):
            k = f'history_{R}'
            new_history[k] = torch.cat((old_history[k], response[:, :, i]), dim=1)[:, -max_H:]
        if update_current:
            self.current_observation['user_history'] = new_history
        return {'slate': rec_list, 'updated_observation': {
            'user_profile': deepcopy(self.current_observation['user_profile']),
            'user_history': deepcopy(new_history)}}

    def sample_new_batch_from_reader(self):
        """
        @output
        - sample_info: see BaseRLEnvironment.get_observation_from_batch@input - sample_batch
        """
        new_sample_flag = False
        try:
            sample_info = next(self.batch_iter)
            if sample_info['user_profile'].shape[0] != self.episode_batch_size:
                new_sample_flag = True
        except:
            new_sample_flag = True
        if new_sample_flag:
            self.batch_iter = iter(DataLoader(self.reader, batch_size=self.episode_batch_size, shuffle=True,
                                              pin_memory=True, num_workers=8))
            sample_info = next(self.batch_iter)
        return sample_info
    
    def reset_new_user(self):
        """
        @output
        - sample_info: see BaseRLEnvironment.get_observation_from_batch@input - sample_batch
        """
        new_sample_flag = False
        try:
            sample_info = next(self.batch_iter)
            if sample_info['user_profile'].shape[0] != self.episode_batch_size:
                new_sample_flag = True
        except:
            new_sample_flag = True
        if new_sample_flag:
            self.batch_iter = iter(DataLoader(self.reader, batch_size=self.episode_batch_size, shuffle=True,
                                              pin_memory=True, num_workers=8))
            sample_info = next(self.batch_iter)
        

    def stop(self):
        self.batch_iter = None

    def get_new_iterator(self, B):
        return iter(DataLoader(self.reader, batch_size=B, shuffle=True,
                               pin_memory=True, num_workers=8))

    def create_observation_buffer(self, buffer_size):
        """
        @input:
        - buffer_size: L, scalar
        @output:
        - observation: {'user_profile': {'user_id': (L,),
                                         'uf_{feature_name}': (L, feature_dim)},
                        'user_history': {'history': (L, max_H),
                                         'history_if_{feature_name}': (L, max_H * feature_dim),
                                         'history_{response}': (L, max_H),
                                         'history_length': (L,)}}
        """
        observation = {'user_profile': {'user_id': torch.zeros(buffer_size).to(torch.long).to(self.device)},
                       'user_history': {
                           'history': torch.zeros(buffer_size, self.max_hist_len).to(torch.long).to(self.device),
                           'history_length': torch.zeros(buffer_size).to(torch.long).to(self.device), 
                           'user_pop_prefer': torch.zeros(buffer_size).to(torch.float).to(self.device), 
                           }}
        for f, f_dim in self.observation_space['user_feature_dims'].items():
            observation['user_profile'][f'uf_{f}'] = torch.zeros(buffer_size, f_dim).to(torch.float).to(self.device)
        for f, f_dim in self.observation_space['item_feature_dims'].items():
            observation['user_history'][f'history_if_{f}'] = torch.zeros(buffer_size, f_dim * self.max_hist_len) \
                .to(torch.float).to(self.device)
        for f in self.observation_space['feedback_type']:
            observation['user_history'][f'history_{f}'] = torch.zeros(buffer_size, self.max_hist_len) \
                .to(torch.float).to(self.device)
        return observation

    def get_report(self, smoothness=10):
        return {k: np.mean(v[-smoothness:]) for k, v in self.env_history.items()}


import argparse

if __name__ == '__main__':
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--env_class', type=str, required=False, help='Environment class.',
                             default='KREnvironment_WholeSession_GPU')
    # init_parser.add_argument('--policy_class', type=str, required=True, help='Policy class')
    # init_parser.add_argument('--critic_class', type=str, required=True, help='Critic class')
    # init_parser.add_argument('--agent_class', type=str, required=True, help='Learning agent class')
    # init_parser.add_argument('--buffer_class', type=str, required=True, help='Buffer class.')

    initial_args, _ = init_parser.parse_known_args()
    envClass = eval(initial_args.env_class)
    print(envClass)
    print(initial_args)
