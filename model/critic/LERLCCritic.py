import pandas as pd
import requests
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import json

from collections import deque
from model.components import DNN
from utils import get_regularization
from concurrent.futures import ThreadPoolExecutor

class LERLCCritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        """
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128],
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.2,
                            help='dropout rate in deep layers')
        return parser

    def __init__(self, args, environment, policy):
        super().__init__()
        self.device = args.device
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        self.effect_action_dim = policy.effect_action_dim
        path = f"dataset/{args.dataset}/"
        item_types_np = pd.read_csv(path + 'item_types.csv').to_numpy().astype(np.float32)
        self.item_types = torch.tensor(item_types_np, dtype=torch.float32).to(self.device)
        self.trans_cat = args.trans_cat == 1
        if self.trans_cat:
            with open(path + 'cat2id.json', 'r', encoding='utf-8') as f:
                self.cat2id = json.load(f)
            with open(path + 'id2cat.json', 'r', encoding='utf-8') as f:
                self.id2cat = json.load(f)
        self.llm_action_dim = np.unique(item_types_np).shape[0]
        self.l_net = DNN(self.state_dim + 0, args.critic_hidden_dims, 1,
                         dropout_rate=args.critic_dropout_rate, do_batch_norm=True)
        self.reflection_buffer = deque(maxlen=200)

    def forward(self, feed_dict):
        """
        @input:
        - feed_dict: {'state_emb': (B, state_dim), 'action_emb': (B, action_dim)}
        """

        l_state_emb = feed_dict['state']
        L_V = self.l_net(torch.cat([l_state_emb], dim=1)).view(-1)
        l_reg = get_regularization(self.l_net)
        return {'v': L_V, 'reg': l_reg}

    def reflect(self, feed_dict, model="llama3", host="http://0.0.0.0:601"):
        prompt_template = """\
[Candidate Categories]
Available categories for selection: {candidate_categories}
        
[User Interaction Summary]
The user's interactions during this session:
{interaction_history}
Session statistics: Total steps = {total_step}, Long-term satisfaction = {cumulative_reward}

[Your Task]
Analyze the session in terms of:
1.Repetition Effects: negative impact of repeated exposure
2.Actionable Insights: strategies to improve long-term satisfaction

[Answer]
Use natural language. Keep total length under 20 words. No intro or conclusion.
"""

        def post_request(url, payload):
            res = requests.post(url, json=payload)
            llm_output = res.json()['response']
            return llm_output
        history = feed_dict['history'] - 1
        total_steps = feed_dict['total_step']
        single_step_rewards = feed_dict['single_step_reward']
        if not self.trans_cat:
            candidate_categories = [i for i in range(self.llm_action_dim)]
        else:
            candidate_categories = [k for k in self.cat2id.keys()]
        total = int(history.shape[0] / 3)
        if history.shape[0] % 3 != 0:
            total += 1
        for j in range(total):
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for k in range(min(3, history.shape[0] - j * 3)):
                    rewards = single_step_rewards[j * 3 + k]
                    total_step = int(total_steps[j * 3 + k]) * self.effect_action_dim
                    categories = self.item_types[history[j * 3 + k]].reshape(-1)
                    categories = categories[-total_step:].reshape(-1).long().tolist()
                    if self.trans_cat:
                        categories = [self.id2cat[str(c)] for c in categories]
                    cul_reward = torch.sum(rewards[-total_step:]).item() if total_step != 0 else 0
                    interaction_lines = f"Interaction Category: {categories}, User Satisfaction: {rewards[-total_step:].long().tolist()}" if total_step != 0 else "The user has no interaction history yet."

                    prompt = prompt_template.format(
                        interaction_history=interaction_lines,
                        cumulative_reward=cul_reward,
                        total_step=total_step,
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
                    reflection = futures[k].result()
                    self.reflection_buffer.append({
                        "cul_reward": cul_reward,
                        "text": reflection
                    })

    def sample_reflections(self, k=3):
        if len(self.reflection_buffer) == 0:
            return []
        rewards = np.array([entry["cul_reward"] for entry in self.reflection_buffer])
        rewards = np.exp(rewards * 0.1) 
        probs = rewards / rewards.sum()

        sampled = np.random.choice(len(self.reflection_buffer),
                                   size=min(k, len(self.reflection_buffer)),
                                   replace=False,
                                   p=probs)
        return [self.reflection_buffer[i]["text"] for i in sampled]

