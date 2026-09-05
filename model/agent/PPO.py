import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.BaseOnPolicyRLAgent import BaseOnPolicyRLAgent
import pandas as pd
from torch import nn


class PPO(BaseOnPolicyRLAgent):
    @staticmethod
    def parse_model_args(parser):
        """
        args:
        - critic_lr
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - reward_func
            - n_iter
            - train_every_n_step
            - start_policy_train_at_step
            - initial_epsilon
            - final_epsilon
            - elbow_epsilon
            - explore_rate
            - do_explore_in_train
            - check_episode
            - save_episode
            - save_path
            - actor_lr
            - actor_decay
            - batch_size
        """
        parser = BaseOnPolicyRLAgent.parse_model_args(parser)
        parser.add_argument('--critic_lr', type=float, default=1e-4,
                            help='decay rate for critic')
        parser.add_argument('--critic_decay', type=float, default=1e-4,
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01,
                            help='mitigation factor')
        parser.add_argument('--train_epoch_num', type=int, default=4,
                            help='train epoch num')
        parser.add_argument('--eps_clip', type=float, default=0.8,
                            help='eps_clip')
        return parser

    def __init__(self, *input_args):
        """
        components:
        - critic
        - critic_optimizer
        - actor_target
        - critic_target
        - components from BaseRLAgent:
            - env
            - actor
            - actor_optimizer
            - buffer
            - exploration_scheduler
            - registered_models
        """
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)

        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef
        self.train_epoch_num = args.train_epoch_num
        self.eps_clip = args.eps_clip
        self.MseLoss = nn.MSELoss()

        # models
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor.set_init_var()
        self.actor_target.set_init_var()

        # controller
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.action_layer.parameters(), 'lr': args.actor_lr},
            {'params': self.actor.user_encoder.parameters(), 'lr': args.actor_lr},
            {'params': self.critic.parameters(), 'lr': args.critic_lr},
            {'params': self.actor.cov_mat, 'lr': args.actor_lr / 10}
        ])
        self.do_actor_update = True
        self.do_critic_update = True

        # register models that will be saved
        self.registered_models.append((self.actor, self.optimizer, ""))
        self.registered_models.append((self.critic, self.optimizer, ""))

    def setup_monitors(self):
        """
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        """
        super().setup_monitors()
        self.training_history.update({'actor_loss': [], 'critic_loss': [], 'dist_entropy_loss': [], 'V': [], 'next_V': []})

    def step_train(self):
        """
        @process:
        - get sample
        - calculate V'(s_{t+1}) and V(s_t)
        - critic loss: TD error loss
        - critic optimization
        - actor loss: ratios * advantages maximization
        - actor optimization
        """
        observation, policy_output, user_feedback, done_mask, n_observation = self.buffer.sample()
        #print(f'mush{done_mask}')
        old_log_prob = policy_output['action_log_prob'].view(-1)
        reward = user_feedback['reward'].view(-1)
        total_len = policy_output['state'].shape[0]
        is_train = True
        # Optimize policy for K epochs
        for _ in range(self.train_epoch_num):
            idxes = np.arange(int((total_len - 1) / self.batch_size))
            np.random.shuffle(idxes)
            for i in idxes:
                # Evaluating old actions and values
                start_idx, end_idx = i * self.batch_size, min((i + 1) * self.batch_size, total_len - 1)
                idx = torch.arange(int(start_idx), int(end_idx)).to(self.device)
                current_observation = {}
                t = observation['user_profile']
                current_observation['user_profile'] = {k: v[idx] for k, v in observation['user_profile'].items()}
                current_observation['user_history'] = {k: v[idx] for k, v in observation['user_history'].items()}
                current_policy_output = {k: v[idx] for k, v in policy_output.items()}
                current_old_log_prob = old_log_prob[idx]
                current_done = done_mask[idx]
                current_reward = reward[idx]
                next_observation = {}
                next_observation['user_profile'] = {k: v[idx] for k, v in n_observation['user_profile'].items()}
                next_observation['user_history'] = {k: v[idx] for k, v in n_observation['user_history'].items()}

                log_prob, dist_entropy = self.actor.evaluate(current_policy_output)
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(log_prob - current_old_log_prob.detach())

                # match state_values tensor dimensions with rewards tensor
                current_critic_output = self.apply_critic(current_observation, current_policy_output, self.critic)
                current_state_values = current_critic_output['v']

                # Compute the target V value
                next_policy_output = self.apply_policy(next_observation, self.actor_target, 0., False, is_train)
                target_critic_output = self.apply_critic(next_observation, next_policy_output, self.critic_target)
                next_state_values = target_critic_output['v']
                target_state_values = self.gamma * torch.squeeze(next_state_values) * (~current_done) + current_reward

                # Finding Surrogate Loss
                advantages = target_state_values - current_state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
                actor_loss = - torch.min(surr1, surr2)
                critic_loss = self.MseLoss(current_state_values, target_state_values) * 0.5
                dist_entropy_loss = -dist_entropy * 0.001
                loss = actor_loss + critic_loss + dist_entropy_loss

                # take gradient step
                self.optimizer.zero_grad()
                with torch.autograd.detect_anomaly():
                    loss.mean().backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=20, norm_type=2)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=20, norm_type=2)
                self.optimizer.step()
                loss_dict = {'actor_loss': actor_loss.mean().item(),
                             'critic_loss': critic_loss.mean().item(),
                             'dist_entropy_loss': dist_entropy_loss.mean().item(),
                             'V': torch.mean(current_state_values).item(),
                             'next_V': torch.mean(next_state_values).item()
                             }

        # Copy new weights into old policy
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for k in loss_dict:
            if k in self.training_history:
                try:
                    self.training_history[k].append(loss_dict[k].item())
                except:
                    self.training_history[k].append(loss_dict[k])

        return loss_dict

    def apply_policy(self, observation, actor, *policy_args):
        """
        @input:
        - observation:{'user_profile':{
                           'user_id': (B,)
                           'uf_{feature_name}': (B,feature_dim), the user features}
                       'user_history':{
                           'history': (B,max_H)
                           'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
        - actor: the actor model
        - epsilon: scalar
        - do_explore: boolean
        - is_train: boolean
        @output:
        - policy_output
        """
        epsilon = policy_args[0]
        do_explore = policy_args[1]
        is_train = policy_args[2]
        input_dict = {'observation': observation,
                      # 默认候选集为整个物品集合
                      'candidates': self.env.get_candidate_info(observation),
                      'epsilon': epsilon,
                      'do_explore': do_explore,
                      'is_train': is_train,
                      'batch_wise': False}
        out_dict = self.actor(input_dict)
        return out_dict

    def apply_critic(self, observation, policy_output, critic):
        feed_dict = {'state': policy_output['state']}
        return critic(feed_dict)

    # def evaluate(self, observation, policy_output):
    #     return self.actor_target.evaluate(observation, policy_output)

    def save(self):
        super().save()

    def load(self):
        super().load()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
