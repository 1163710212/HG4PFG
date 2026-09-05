import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from model.agent.BaseOnPolicyRLAgent import BaseOnPolicyRLAgent
from model.policy.HRL4PFGPolicy import enable_safe_transformer_attention


class HRL4PFGPPO(BaseOnPolicyRLAgent):
    """Two-timescale HRL4PFG actor-critic optimized with clipped PPO."""

    @staticmethod
    def parse_model_args(parser):
        parser = BaseOnPolicyRLAgent.parse_model_args(parser)
        parser.add_argument('--critic_lr', type=float, default=1e-3)
        parser.add_argument('--critic_decay', type=float, default=1e-5)
        parser.add_argument(
            '--target_mitigate_coef', type=float, default=0.01,
            help='Polyak coefficient for rollout and value target networks',
        )
        parser.add_argument('--train_epoch_num', type=int, default=4)
        parser.add_argument('--eps_clip', type=float, default=0.8)
        parser.add_argument('--entropy_coef', type=float, default=5e-4)
        parser.add_argument(
            '--high_entropy_coef', type=float, default=None,
            help='macro-policy entropy coefficient',
        )
        parser.add_argument(
            '--low_entropy_coef', type=float, default=None,
            help='micro-policy categorical entropy coefficient',
        )
        parser.add_argument('--value_loss_coef', type=float, default=1.0)
        parser.add_argument('--max_grad_norm', type=float, default=10.0)
        parser.add_argument(
            '--lambda_fairness', type=float, default=0.1,
            help='lambda_f for the entropy-normalized fairness reward',
        )
        parser.add_argument(
            '--lambda_guide', type=float, default=0.1,
            help='lambda_g for the low-level guidance reward',
        )
        parser.add_argument(
            '--subgoal_interval', type=int, default=4,
            help='M: low-level steps governed by one macro target',
        )
        parser.add_argument(
            '--low_level_only', action='store_true',
            help='ablation: optimize only the low-level PPO and value function',
        )
        parser.add_argument(
            '--train_low_user_encoder', action='store_true',
            help=(
                'ablation: recompute low-level states from buffered '
                'observations so l_user_encoder and preference_layer receive '
                'PPO/value gradients'
            ),
        )
        return parser

    def __init__(self, args, environment, actor, critic, buffer):
        if args.subgoal_interval <= 0:
            raise ValueError('subgoal_interval must be positive')
        if not 0.0 < args.target_mitigate_coef <= 1.0:
            raise ValueError('target_mitigate_coef must be in (0, 1]')
        if not 0.0 < args.eps_clip < 1.0:
            raise ValueError('eps_clip must be in (0, 1)')
        if args.lambda_fairness < 0.0 or args.lambda_guide < 0.0:
            raise ValueError('lambda_fairness and lambda_guide must be non-negative')
        if args.entropy_coef < 0.0:
            raise ValueError('entropy_coef must be non-negative')
        if args.high_entropy_coef is not None and args.high_entropy_coef < 0.0:
            raise ValueError('high_entropy_coef must be non-negative')
        if args.low_entropy_coef is not None and args.low_entropy_coef < 0.0:
            raise ValueError('low_entropy_coef must be non-negative')

        super().__init__(args, environment, actor, buffer)
        # BaseOnPolicyRLAgent truncates this horizon to int; make smoke runs
        # with n_iter=1 well-defined without changing the shared base class.
        schedule_horizon = max(1, int(sum(args.n_iter) * args.elbow_epsilon))
        self.exploration_scheduler = utils.LinearScheduler(
            schedule_horizon,
            args.final_epsilon,
            initial_p=args.initial_epsilon,
        )

        self.critic = critic
        self.critic_target = copy.deepcopy(critic)
        self.tau = args.target_mitigate_coef
        self.train_epoch_num = args.train_epoch_num
        self.eps_clip = args.eps_clip
        self.high_entropy_coef = (
            args.entropy_coef
            if args.high_entropy_coef is None
            else args.high_entropy_coef
        )
        self.low_entropy_coef = (
            args.entropy_coef
            if args.low_entropy_coef is None
            else args.low_entropy_coef
        )
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm
        self.lambda_fairness = args.lambda_fairness
        self.lambda_guide = args.lambda_guide
        self.subgoal_interval = args.subgoal_interval
        self.low_level_only = getattr(args, 'low_level_only', False)
        self.train_low_user_encoder = bool(
            getattr(args, 'train_low_user_encoder', False)
        )
        if self.low_level_only:
            for high_module in (
                self.actor.h_user_encoder,
                self.actor.h_action_layer,
                self.actor.h_action_log_std_layer,
                self.critic.h_net,
            ):
                high_module.requires_grad_(False)
        print(
            'HRL4PFG PPO training mode: {0}'.format(
                'low-level-only' if self.low_level_only else 'joint high+low'
            )
        )
        print(
            'HRL4PFG low user encoder: {0}'.format(
                'trainable from buffered observations'
                if self.train_low_user_encoder
                else 'original detached rollout-state path'
            )
        )

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()
        self.critic_target.eval()
        for target_model in (self.actor_target, self.critic_target):
            for parameter in target_model.parameters():
                parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            [
                {
                    'params': self.actor.parameters(),
                    'lr': args.actor_lr,
                    'weight_decay': args.actor_decay,
                },
                {
                    'params': self.critic.parameters(),
                    'lr': args.critic_lr,
                    'weight_decay': args.critic_decay,
                },
            ]
        )
        self.registered_models.append((self.actor, self.optimizer, '_actor'))
        self.registered_models.append((self.critic, self.optimizer, '_critic'))

    @staticmethod
    def _copy_observation_rows(target, source, mask):
        for group in ('user_profile', 'user_history'):
            for key in target[group]:
                target[group][key][mask] = source[group][key][mask].detach()

    @staticmethod
    def _slice_observation(observation, indices):
        return {
            group: {key: value[indices] for key, value in observation[group].items()}
            for group in ('user_profile', 'user_history')
        }

    def _shape_hierarchical_rewards(
        self, accuracy_reward, guidance_reward, fairness_reward
    ):
        low_reward = accuracy_reward + self.lambda_guide * guidance_reward
        high_step_reward = (
            accuracy_reward + self.lambda_fairness * fairness_reward
        )
        return low_reward, high_step_reward

    def _reset_hierarchy_state(self, environment):
        batch_size = environment.episode_batch_size
        goal_dim = self.actor.goal_dim
        self.active_goal = torch.zeros(
            batch_size, goal_dim, device=self.device
        )
        self.active_goal_snap_distance = torch.zeros(
            batch_size, device=self.device
        )
        self.goal_active = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )

        self.pending_high_observation = environment.create_observation_buffer(
            batch_size
        )
        self.pending_high_prototype = torch.zeros(
            batch_size, goal_dim, device=self.device
        )
        self.pending_high_log_prob = torch.zeros(batch_size, device=self.device)
        self.pending_high_reward = torch.zeros(batch_size, device=self.device)
        self.pending_high_duration = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )

    def setup_monitors(self):
        super().setup_monitors()
        self.training_history.update(
            {
                'h_actor_loss': [],
                'h_critic_loss': [],
                'l_actor_loss': [],
                'l_critic_loss': [],
                'h_entropy': [],
                'h_entropy_per_dim': [],
                'l_entropy': [],
                'critic_loss': [],
                'dist_entropy_loss': [],
            }
        )
        self.eval_history.update(
            {
                'guidance_reward': [],
                'fairness_reward': [],
                'high_step_reward': [],
                'low_step_reward': [],
                'goal_snap_distance': [],
                'goal_std': [],
                'long_tail_ratio': [],
            }
        )
        self._reset_hierarchy_state(self.env)

    def apply_policy(self, observation, actor, *policy_args):
        do_explore = policy_args[1]
        environment = policy_args[3] if len(policy_args) > 3 else self.env
        batch_size = observation['user_profile']['user_id'].shape[0]
        if self.active_goal.shape[0] != batch_size:
            self._reset_hierarchy_state(environment)

        current_step = observation.get('current_step', self.current_step).long()
        new_goal_mask = (~self.goal_active) | (
            current_step % self.subgoal_interval == 0
        )
        output = actor(
            {
                'observation': observation,
                'candidates': environment.get_candidate_info(observation),
                'do_explore': do_explore,
                'new_goal_mask': new_goal_mask,
                'previous_goal': self.active_goal,
                'previous_goal_snap_distance': self.active_goal_snap_distance,
            }
        )
        self.active_goal = output['goal'].detach().clone()
        self.active_goal_snap_distance = output[
            'goal_snap_distance'
        ].detach().clone()
        self.goal_active[:] = True
        output['new_goal_mask'] = new_goal_mask
        return output

    def run_episode_step(self, *episode_args):
        _, epsilon, observation, do_buffer_update, do_explore = episode_args
        self.epsilon = epsilon
        pre_step = self.current_step.long().clone()
        observation['current_step'] = pre_step

        with torch.no_grad():
            policy_output = self.apply_policy(
                observation, self.actor_target, epsilon, do_explore, False
            )
            new_goal_mask = policy_output['new_goal_mask']
            if torch.any(new_goal_mask):
                self._copy_observation_rows(
                    self.pending_high_observation, observation, new_goal_mask
                )
                self.pending_high_prototype[new_goal_mask] = policy_output[
                    'h_action'
                ][new_goal_mask]
                self.pending_high_log_prob[new_goal_mask] = policy_output[
                    'h_action_log_prob'
                ][new_goal_mask]
                self.pending_high_reward[new_goal_mask] = 0.0
                self.pending_high_duration[new_goal_mask] = 0

            action = policy_output['indices']
            new_observation, user_feedback, update_info, current_step = self.env.step(
                {'action': action}
            )
            # The environment has already inserted replacement users into
            # new_observation.  updated_observation is the true s_{t+1} for
            # users whose episodes ended at this transition.
            terminal_next_observation = update_info['updated_observation']
            accuracy_reward = self.get_reward(user_feedback)
            user_feedback['reward'] = accuracy_reward
            done_mask = user_feedback['done'].bool()

            next_preference = self.actor_target.encode_preference(
                terminal_next_observation
            )
            distance_before = torch.norm(
                policy_output['preference'] - policy_output['goal'],
                p=2,
                dim=-1,
            )
            distance_after = torch.norm(
                next_preference - policy_output['goal'], p=2, dim=-1
            )
            guidance_reward = distance_before - distance_after
            fairness_reward = policy_output['fairness_reward']
            low_reward, high_step_reward = self._shape_hierarchical_rewards(
                accuracy_reward,
                guidance_reward,
                fairness_reward,
            )
            macro_discount = accuracy_reward.new_full(
                accuracy_reward.shape, self.gamma
            ).pow(self.pending_high_duration)
            self.pending_high_reward += macro_discount * high_step_reward
            self.pending_high_duration += 1

            interval_end = (pre_step + 1) % self.subgoal_interval == 0
            macro_end = interval_end | done_mask

            # The low-level Bellman chain continues across macro-target
            # boundaries.  At such a boundary, bootstrap from the deterministic
            # target-policy goal for s_{t+1}; only a real user exit is terminal.
            next_goal = policy_output['goal'].detach().clone()
            next_goal_mask = interval_end & (~done_mask)
            if torch.any(next_goal_mask):
                next_goal_indices = torch.where(next_goal_mask)[0]
                next_goal_observation = self._slice_observation(
                    terminal_next_observation, next_goal_indices
                )
                next_goal[next_goal_indices] = (
                    self.actor_target.deterministic_snapped_goal(
                        next_goal_observation,
                        self.env.get_candidate_info(next_goal_observation),
                    )
                )
            if do_buffer_update:
                self.buffer.append_low(
                    observation,
                    terminal_next_observation,
                    policy_output['goal'],
                    next_goal,
                    policy_output['l_state'],
                    action,
                    policy_output['l_action_log_prob'],
                    low_reward,
                    done_mask,
                )
                self.buffer.append_high(
                    self.pending_high_observation,
                    terminal_next_observation,
                    self.pending_high_prototype,
                    self.pending_high_log_prob,
                    self.pending_high_reward,
                    done_mask,
                    self.pending_high_duration,
                    torch.where(macro_end)[0],
                )

            self.goal_active[macro_end] = False
            self.current_step = current_step.long()

            self.current_sum_reward += accuracy_reward
            if torch.any(done_mask):
                finished_returns = self.current_sum_reward[done_mask]
                self.eval_history['avg_total_reward'].append(
                    finished_returns.mean().item()
                )
                self.eval_history['max_total_reward'].append(
                    finished_returns.max().item()
                )
                self.eval_history['min_total_reward'].append(
                    finished_returns.min().item()
                )
                self.current_sum_reward[done_mask] = 0.0

            self.eval_history['avg_reward'].append(accuracy_reward.mean().item())
            self.eval_history['reward_variance'].append(
                torch.var(accuracy_reward, unbiased=False).item()
            )
            self.eval_history['guidance_reward'].append(
                guidance_reward.mean().item()
            )
            self.eval_history['fairness_reward'].append(
                fairness_reward.mean().item()
            )
            self.eval_history['high_step_reward'].append(
                high_step_reward.mean().item()
            )
            self.eval_history['low_step_reward'].append(low_reward.mean().item())
            for key in ('goal_snap_distance', 'goal_std'):
                self.eval_history[key].append(policy_output[key].mean().item())
            self.eval_history['long_tail_ratio'].append(
                policy_output['unpopular_item_ratio'].mean().item()
            )
            for response_index, response_name in enumerate(self.env.response_types):
                self.eval_history[response_name + '_rate'].append(
                    user_feedback['immediate_response'][:, :, response_index]
                    .mean()
                    .item()
                )

        return new_observation

    @staticmethod
    def _minibatches(size, batch_size, device):
        if size <= 0:
            return
        permutation = torch.randperm(size, device=device)
        for start in range(0, size, batch_size):
            yield permutation[start:start + batch_size]

    def _low_batch_terms(self, data, indices):
        observation = self._slice_observation(
            data['observation'], indices
        )
        next_observation = self._slice_observation(
            data['next_observation'], indices
        )
        goal = data['goal'][indices]
        next_goal = data['next_goal'][indices]
        if self.train_low_user_encoder:
            # Ablation-only path: rebuild the state with the live actor so PPO
            # and value losses update l_user_encoder + preference_layer.
            preference = self.actor.encode_preference(observation)
            low_state = torch.cat((preference, goal.detach()), dim=-1)
        else:
            # Original-model path remains byte-for-byte equivalent: consume
            # the detached rollout state stored by HRL4PFGBuffer.
            low_state = data['low_state'][indices]
        action = data['action'][indices]
        old_log_prob = data['old_log_prob'][indices]
        reward = data['reward'][indices]
        done = data['done'][indices]

        log_prob, entropy, low_state = self.actor.evaluate_low(
            low_state,
            action,
            detach_state=not self.train_low_user_encoder,
        )
        current_value = self.critic.low_value(low_state)
        with torch.no_grad():
            next_preference = self.actor_target.encode_preference(next_observation)
            next_low_state = torch.cat((next_preference, next_goal), dim=-1)
            next_value = self.critic_target.low_value(next_low_state)
            target_value = reward + self.gamma * next_value * (~done).float()
            advantage = target_value - current_value.detach()

        # Keep the PPO ratio definition identical to HPPO.step_train().
        ratio = torch.exp(log_prob - old_log_prob.detach())
        surrogate_1 = ratio * advantage
        surrogate_2 = torch.clamp(
            ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip
        ) * advantage
        actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
        critic_loss = F.mse_loss(current_value, target_value)
        entropy_mean = entropy.mean()
        return actor_loss, critic_loss, entropy_mean

    def _high_batch_terms(self, data, indices):
        observation = self._slice_observation(data['observation'], indices)
        next_observation = self._slice_observation(
            data['next_observation'], indices
        )
        prototype = data['prototype'][indices]
        old_log_prob = data['old_log_prob'][indices]
        reward = data['reward'][indices]
        done = data['done'][indices]
        log_prob, entropy, high_state = self.actor.evaluate_high(
            observation, prototype
        )
        current_value = self.critic.high_value(high_state)
        with torch.no_grad():
            next_high_state = self.actor_target.encode_high_state(next_observation)
            next_value = self.critic_target.high_value(next_high_state)
            # One macro Bellman step spans the completed target interval.
            macro_discount = reward.new_full(
                reward.shape, self.gamma
            ).pow(data['duration'][indices])
            target_value = (
                reward + macro_discount * next_value * (~done).float()
            )
            advantage = target_value - current_value.detach()

        # Keep the PPO ratio definition identical to HPPO.step_train().
        ratio = torch.exp(log_prob - old_log_prob.detach())
        surrogate_1 = ratio * advantage
        surrogate_2 = torch.clamp(
            ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip
        ) * advantage
        actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
        critic_loss = F.mse_loss(current_value, target_value)
        entropy_mean = entropy.mean()
        entropy_per_dimension = entropy_mean / float(self.actor.goal_dim)
        return actor_loss, critic_loss, entropy_mean, entropy_per_dimension

    @staticmethod
    def _mean_or_zero(values):
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _soft_update(source, target, tau):
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(
                    source_parameter, alpha=tau
                )
            for source_buffer, target_buffer in zip(
                source.buffers(), target.buffers()
            ):
                target_buffer.copy_(source_buffer)

    def step_train(self):
        """Apply clipped PPO jointly, or only at the low level for ablation."""
        low_data = self.buffer.low_data()
        high_data = None if self.low_level_only else self.buffer.high_data()
        metrics = {
            'h_actor_loss': [],
            'h_critic_loss': [],
            'l_actor_loss': [],
            'l_critic_loss': [],
            'h_entropy': [],
            'h_entropy_per_dim': [],
            'l_entropy': [],
            'critic_loss': [],
            'dist_entropy_loss': [],
        }

        self.actor.train()
        self.critic.train()
        self.actor_target.eval()
        self.critic_target.eval()
        for _ in range(self.train_epoch_num):
            low_batches = list(
                self._minibatches(
                    self.buffer.low_size, self.batch_size, self.device
                )
            )
            high_batches = (
                []
                if self.low_level_only
                else list(
                    self._minibatches(
                        self.buffer.high_size, self.batch_size, self.device
                    )
                )
            )
            batch_count = max(len(low_batches), len(high_batches))
            for batch_index in range(batch_count):
                actor_loss = torch.zeros((), device=self.device)
                critic_loss = torch.zeros((), device=self.device)
                dist_entropy_loss = torch.zeros((), device=self.device)

                if batch_index < len(low_batches):
                    l_actor_loss, l_critic_loss, l_entropy = (
                        self._low_batch_terms(
                            low_data, low_batches[batch_index]
                        )
                    )
                    actor_loss = actor_loss + l_actor_loss
                    critic_loss = critic_loss + l_critic_loss
                    dist_entropy_loss = (
                        dist_entropy_loss
                        - self.low_entropy_coef * l_entropy
                    )
                    metrics['l_actor_loss'].append(l_actor_loss.item())
                    metrics['l_critic_loss'].append(l_critic_loss.item())
                    metrics['l_entropy'].append(l_entropy.item())

                if batch_index < len(high_batches):
                    (
                        h_actor_loss,
                        h_critic_loss,
                        h_entropy,
                        h_entropy_per_dim,
                    ) = self._high_batch_terms(
                        high_data, high_batches[batch_index]
                    )
                    actor_loss = actor_loss + h_actor_loss
                    critic_loss = critic_loss + h_critic_loss
                    dist_entropy_loss = (
                        dist_entropy_loss
                        - self.high_entropy_coef * h_entropy
                    )
                    metrics['h_actor_loss'].append(h_actor_loss.item())
                    metrics['h_critic_loss'].append(h_critic_loss.item())
                    metrics['h_entropy'].append(h_entropy.item())
                    metrics['h_entropy_per_dim'].append(
                        h_entropy_per_dim.item()
                    )

                # Same combined objective and update order as HPPO.step_train:
                # high actor + low actor + both critics + both entropies.
                loss = (
                    actor_loss
                    + self.value_loss_coef * critic_loss
                    + dist_entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=self.max_grad_norm,
                    norm_type=2,
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=self.max_grad_norm,
                    norm_type=2,
                )
                self.optimizer.step()
                metrics['critic_loss'].append(critic_loss.item())
                metrics['dist_entropy_loss'].append(
                    dist_entropy_loss.item()
                )

        self._soft_update(self.actor, self.actor_target, self.tau)
        self._soft_update(self.critic, self.critic_target, self.tau)
        self.actor_target.eval()
        self.critic_target.eval()

        loss_dict = {
            key: self._mean_or_zero(values) for key, values in metrics.items()
        }
        for key, value in loss_dict.items():
            self.training_history[key].append(value)
        return loss_dict

    def _prepare_test_repeat(self, test_environment):
        enable_safe_transformer_attention(
            test_environment.immediate_response_model
        )
        self._reset_hierarchy_state(test_environment)

    def load(self):
        super().load()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.eval()
        self.critic_target.eval()
