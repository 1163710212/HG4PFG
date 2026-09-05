import torch

from model.policy.OneStagePolicy import OneStagePolicy
from model.components import DNN
from model.score_func import *
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import Multinomial
import torch.nn as nn
import requests
import ast
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json

class OneStagePolicy_LERLC(OneStagePolicy):
    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - from OneStagePolicy:
            - state_encoder_feature_dim
            - state_encoder_attn_n_head
            - state_encoder_hidden_dims
            - state_encoder_dropout_rate
        """
        parser = OneStagePolicy.parse_model_args(parser)
        parser.add_argument('--policy_action_hidden', type=int, nargs='+', default=[128],
                            help='hidden dim of the action net')
        parser.add_argument('--action_std_init', type=float, default=0.8,
                            help='init action std')
        parser.add_argument('--has_continuous_action_space', type=bool, default=True,
                            help='action type')
        parser.add_argument('--trans_cat', type=int, default=0,
                            help='translate category ids with dataset cat2id/id2cat mappings')
        return parser

    def __init__(self, args, environment):
        """
        action_space = {'item_id': ('nominal', stats['n_item']),
                        'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive']),
                            'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}
        """
        super().__init__(args, environment)
        # action is the set of parameters of linear mapping [item_dim, 1]
        path = f"dataset/{args.dataset}/"
        item_types_np = pd.read_csv(path + 'item_types.csv').to_numpy().astype(np.float32)
        self.item_types = torch.tensor(item_types_np, dtype=torch.float32).to(self.device)
        self.llm_action_dim = np.unique(item_types_np).shape[0]
        self.hyper_action_dim = self.enc_dim + 1
        self.action_dim = self.hyper_action_dim
        self.effect_action_dim = self.slate_size
        self.has_continuous_action_space = args.has_continuous_action_space
        self.trans_cat = args.trans_cat == 1
        if self.trans_cat:
            with open(path + 'cat2id.json', 'r', encoding='utf-8') as f:
                self.cat2id = json.load(f)
            with open(path + 'id2cat.json', 'r', encoding='utf-8') as f:
                self.id2cat = json.load(f)
        self.action_layer = DNN(self.state_dim, args.policy_action_hidden, self.action_dim,
                                dropout_rate=self.dropout_rate, do_batch_norm=True)
        self.action_std_init = args.action_std_init

    def set_init_var(self):
        action_var = torch.full((self.action_dim,), self.action_std_init * self.action_std_init).to(self.device)
        # self.action_var = nn.Parameter(action_var)
        self.cov_mat = nn.Parameter(torch.diag(action_var))

    def generate_action(self, state_dict, feed_dict):
        """
        List generation provides three main types of exploration:
        * Greedy top-K: no exploration, set do_effect_action_explore=False in args or do_explore=False in feed_dict
        * Categorical sampling: probabilistic exploration, set do_effect_action_explore=True in args,
                                set do_explore=True and epsilon < 1 in feed_dict
        * Uniform sampling : random exploration, set do_effect_action_explore=True in args,
                             set do_explore=True, epsilon > 0 in feed_dict
        * Gaussian sampling on hyper-action: set do_explore=True, epsilon < 1 in feed_dict
        * Uniform sampling on hyper-action: set do_explore=True, epsilon > 0 in feed_dict

        @input:
        - state_dict: {'state': (B, state_dim), ...}
        - feed_dict: same as OneStagePolicy.get_forward@input - feed_dict
        @output:
        - out_dict: {'action': (B, K),
                     'action_log_prb': (B, K),
                     'action_prob': (B, K),
                     'reg': scalar}
        """
        state = state_dict['state']
        candidates = feed_dict['candidates']
        batch_wise = feed_dict['batch_wise']


        llm_action = self.select_category(feed_dict)
        selected_categories = [torch.nonzero(llm_action[i], as_tuple=False).squeeze(1) for i in
                               range(llm_action.size(0))]
        mask = torch.stack([
            torch.isin(self.item_types, selected_categories[i])  # [item_num]
            for i in range(llm_action.size(0))
        ])

        action_mean = self.action_layer(state)
        # cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, torch.abs(self.cov_mat) + 1e-4)

        hyper_action = dist.sample()
        action_log_prob = dist.log_prob(hyper_action)

        candidate_item_enc, reg = self.user_encoder.get_item_encoding(candidates['item_id'],
                                                                      {k[3:]: v for k, v in candidates.items() if
                                                                       k != 'item_id'},
                                                                      B if batch_wise else 1)
        # (B, L)
        scores = self.get_score(hyper_action, candidate_item_enc, self.enc_dim)
        # top-k selection
        _, indices = torch.topk(scores * mask.squeeze().float(), k=self.slate_size, dim=1)

        reg += self.get_regularization(self.action_layer)
        out_dict = {'action': hyper_action,
                    'llm_action': llm_action,
                    'action_log_prob': action_log_prob,
                    'indices': indices,
                    'reg': reg}
        return out_dict

    def get_score(self, hyper_action, candidate_item_enc, item_dim):
        '''
        Deterministic mapping from hyper-action to effect-action (rec list)
        '''
        # (B, L)
        scores = linear_scorer(hyper_action, candidate_item_enc, item_dim)
        return scores

    def evaluate(self, feed_dict):
        llm_action = feed_dict['llm_action'].view(-1)
        l_state = feed_dict['state'].view(-1, self.state_dim)
        l_action = feed_dict['action'].view(-1, self.action_dim)

        if self.has_continuous_action_space:
            action_mean = self.action_layer(l_state)
            try:
                dist = MultivariateNormal(action_mean, torch.abs(self.cov_mat) + 1e-4)
            except:
                print(nn.functional.relu(self.cov_mat) + 1e-6)
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_value = self.action_layer(l_state)
            dist = Multinomial(self.slate_size, action_prob)
        action_log_prob = dist.log_prob(l_action)
        dist_entropy = dist.entropy()
        return action_log_prob, dist_entropy

    def get_forward(self, feed_dict: dict):
        observation = feed_dict['observation']
        # observation --> user state
        state_dict = self.get_user_state(observation)
        # user state + candidates --> dict(state, prob, action, reg)
        is_train = feed_dict['is_train']
        if is_train:
            out_dict = {}
            out_dict['reg'] = state_dict['reg']
        else:
            out_dict = self.generate_action(state_dict, feed_dict)
            out_dict['reg'] = state_dict['reg'] + out_dict['reg']
        out_dict['state'] = state_dict['state']
        return out_dict

    def forward(self, feed_dict: dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        return out_dict

    def select_category(self, feed_dict, model="llama3", host="http://0.0.0.0:601"):
        """
        observation:{
                'user_profile':{
                    'user_id': (B, 1)
                    'uf_{feature_name}': (B,feature_dim), the user features}
                'user_history':{
                    'history': (B,max_H)
                    'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
        """
        prompt_template = """\
[Candidate Categories]
Available categories for selection: {candidate_categories}

[User Interaction History]
{interaction_history}

[Reflective Summaries]
These summaries are derived from users who ultimately achieved a high long-term satisfaction:  
{reflective_summary}

[Your Task]
You are a recommendation planning assistant. Your task is to choose the next item categories to recommend based on the user's interaction history and reflective summaries. Prioritize categories that are relevant to user interests. The list length must be more than 1. Your goal is to maximize user long-term satisfaction.

[Answer]
Output strictly as a Python list of selected category integers, e.g., [1, 2, 3]. No explanation, prefix, or formatting.
"""     
        print(feed_dict.keys(), feed_dict['observation'].keys())
        rewards = feed_dict['observation']['user_history']['history_is_click']
        current_step = feed_dict['observation']['current_step']
        reflections = feed_dict['reflections']
        # 'history': (B,max_H)
        interactions = feed_dict['observation']['user_history']['history'] - 1
        if not self.trans_cat:
            candidate_categories = [i for i in range(self.llm_action_dim)]
        else:
            candidate_categories = [k for k in self.cat2id.keys()]

        def post_request(url, payload):
            while (True):
                llm_output = requests.post(url, json=payload).json()['response']
                try:
                    llm_output = ast.literal_eval(llm_output)
                    if isinstance(llm_output, list):
                        break
                except Exception:
                    pass
            return llm_output

        selected_categories = []
        total = int(interactions.shape[0] / 3)
        if interactions.shape[0] % 3 != 0:
            total += 1
        for j in range(total):
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for k in range(min(3, interactions.shape[0] - j * 3)):
                    rwds = rewards[j * 3 + k]
                    step = min(int(current_step[j * 3 + k]), 5) * self.effect_action_dim
                    categories = self.item_types[interactions[j * 3 + k]]
                    categories = categories[-step:].reshape(-1).long().tolist()
                    #print(categories)
                    if self.trans_cat:
                        categories = [self.id2cat[str(c)] for c in categories]
                    interaction_lines = f"Interaction Category: {categories}\nUser Satisfaction: {rwds[-step:].long().tolist()}." if step != 0 else "The user has no interaction history yet."

                    reflection_lines = [f"-User{i}:{r}" for i, r in enumerate(reflections)]

                    prompt = prompt_template.format(
                        interaction_history=interaction_lines,
                        reflective_summary="\n".join(reflection_lines),
                        candidate_categories=candidate_categories
                    )
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    }
                    future = executor.submit(post_request, f"{host}{k+1}/api/generate", payload)
                    futures.append(future)

            for k in range(len(futures)):
                llm_output = futures[k].result()
                selected_category = np.zeros(self.llm_action_dim)
                for c in llm_output:
                    c = int(c)
                    if c < self.llm_action_dim and c > 0:
                        selected_category[c] = 1
                selected_categories.append(selected_category)
        return torch.tensor(selected_categories).long().to(self.device)
