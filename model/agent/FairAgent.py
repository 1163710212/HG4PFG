import copy
import numpy as np
import torch
import torch.nn.functional as F

import utils
from model.agent.BaseRLAgent import BaseRLAgent


class FairAgent(BaseRLAgent):
    """Paper-aligned list-wise adaptation of FairAgent for this simulator.

    The repository environment displays a complete slate in one call, whereas
    Algorithm 1 in the paper selects the slate item by item.  This agent retains
    the per-position rewards and without-replacement action construction, then
    treats the completed slate as one replay transition.
    """

    @staticmethod
    def parse_model_args(parser):
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument(
            '--fair_alpha', type=float, default=2.0,
            help='alpha multiplying the personalized fairness reward',
        )
        parser.add_argument(
            '--fair_beta', type=float, default=0.5,
            help='beta multiplying the new-item exploration reward',
        )
        parser.add_argument(
            '--fair_new_reward_gamma', type=float, default=0.1,
            help='gamma distributing new-item reward between exposure and positive feedback',
        )
        parser.add_argument(
            '--fair_target_update_interval', type=int, default=5,
            help='hard target-network update interval from Algorithm 1',
        )
        parser.add_argument(
            '--fair_ad_weight', type=float, default=0.5,
            help=(
                'weight of the learned slate-balance reward K * (1 - AD); '
                'zero disables AD reward shaping'
            ),
        )
        return parser

    def __init__(self, *input_args):
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)

        self.fair_alpha = args.fair_alpha
        self.fair_beta = args.fair_beta
        self.fair_new_reward_gamma = args.fair_new_reward_gamma
        self.target_update_interval = args.fair_target_update_interval
        self.fair_ad_weight = args.fair_ad_weight
        if self.fair_alpha < 0.0 or self.fair_beta < 0.0:
            raise ValueError('fair_alpha and fair_beta must be non-negative')
        if not 0.0 <= self.fair_new_reward_gamma <= 1.0:
            raise ValueError('fair_new_reward_gamma must be in [0, 1]')
        if self.target_update_interval <= 0:
            raise ValueError('fair_target_update_interval must be positive')
        if self.fair_ad_weight < 0.0:
            raise ValueError('fair_ad_weight must be non-negative')

        self.critic = critic  # Kept only for the shared train_actor_critic interface.
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        for parameter in self.actor_target.parameters():
            parameter.requires_grad_(False)
        self.fair_train_step = 0

    def setup_monitors(self):
        super().setup_monitors()
        self.training_history = {
            'loss': [],
            'Q': [],
            'next_Q': [],
            'target_Q': [],
        }
        self.eval_history.update({
            'objective_reward': [],
            'accuracy_reward': [],
            'fairness_reward': [],
            'new_item_reward': [],
            'new_item_coverage': [],
            'ad_reward': [],
            'slate_ad': [],
            'TGF': [],
            'UNF': [],
        })

    def apply_policy(self, observation, actor, *policy_args):
        epsilon, do_explore, is_train = policy_args
        input_dict = {
            'observation': observation,
            'candidates': self.env.get_candidate_info(observation),
            'epsilon': epsilon,
            'do_explore': do_explore,
            'is_train': is_train,
            'batch_wise': False,
        }
        return actor(input_dict)

    def _reward_components(self, observation, action_indices, user_feedback):
        """Implement Equations (9)--(12) and the user-level UNF objective."""
        try:
            positive_index = self.env.response_types.index(self.actor.fair_positive_feedback)
        except ValueError as error:
            raise ValueError(
                f'Environment has no response {self.actor.fair_positive_feedback!r}'
            ) from error
        try:
            click_index = self.env.response_types.index('is_click')
        except ValueError as error:
            raise ValueError("FairAgent avg_reward requires an 'is_click' response") from error

        immediate_response = user_feedback['immediate_response']
        positive = immediate_response[:, :, positive_index].float()
        click_reward = immediate_response[:, :, click_index].float().mean(dim=1)
        slate_size = action_indices.shape[1]
        rank = torch.arange(1, slate_size + 1, device=self.device).float()
        exposure = 1.0 / torch.log2(rank + 1.0)

        # Equation (11), accumulated over the K sequential choices.
        accuracy_per_position = positive * exposure.view(1, -1)
        accuracy_reward = accuracy_per_position.sum(dim=1)

        # Equation (9).
        is_new = self.actor.new_item_mask[action_indices].float()
        new_per_position = (
            self.fair_new_reward_gamma * is_new
            + (1.0 - self.fair_new_reward_gamma) * is_new * positive
        )
        new_item_reward = new_per_position.sum(dim=1)

        # Equations (4)--(6) and (10).  Equation (6) defines UNF as a
        # squared divergence.  Using its decrease makes the reward positive if
        # and only if the generated prefix moves closer to the user's history.
        history_tgf = self.actor.compute_history_tgf(observation)
        prefix_tgf = self.actor.compute_prefix_tgf(action_indices)
        prefix_unf = (prefix_tgf - history_tgf.unsqueeze(1)).pow(2)
        unf_improvement = prefix_unf[:, :-1] - prefix_unf[:, 1:]
        fair_normalizer = 2.0 / (1.0 + np.tanh(2.0))
        fairness_per_position = fair_normalizer * torch.tanh(unf_improvement)
        fairness_reward = fairness_per_position.sum(dim=1)

        # The simulator's AD objective uses popular/long-tail item labels,
        # whereas the FairAgent paper reward above uses temporal old/new
        # groups.  Expose AD as an environment reward instead of imposing a
        # fixed group quota in the policy.  Scaling by K keeps its magnitude
        # comparable with the other list-wise reward sums.
        item_types = self.env.item_types.reshape(-1)[action_indices].float()
        popular_ratio = item_types.mean(dim=1)
        slate_ad = torch.abs(popular_ratio - (1.0 - popular_ratio))
        ad_reward = slate_size * (1.0 - slate_ad)

        # Equation (12), extended with the learned AD feedback.
        total_reward = (
            accuracy_reward
            + self.fair_alpha * fairness_reward
            + self.fair_beta * new_item_reward
            + self.fair_ad_weight * ad_reward
        )
        return {
            'total': total_reward,
            'click': click_reward,
            'accuracy': accuracy_reward,
            'fairness': fairness_reward,
            'new_item': new_item_reward,
            'new_item_coverage': is_new.mean(dim=1),
            'ad_reward': ad_reward,
            'slate_ad': slate_ad,
            'tgf': prefix_tgf[:, -1],
            'unf': prefix_unf[:, -1],
        }

    def _record_reward_components(self, components):
        mapping = {
            'objective_reward': 'total',
            'accuracy_reward': 'accuracy',
            'fairness_reward': 'fairness',
            'new_item_reward': 'new_item',
            'new_item_coverage': 'new_item_coverage',
            'ad_reward': 'ad_reward',
            'slate_ad': 'slate_ad',
            'TGF': 'tgf',
            'UNF': 'unf',
        }
        for history_key, component_key in mapping.items():
            self.eval_history[history_key].append(
                components[component_key].mean().item()
            )

    def run_episode_step(self, *episode_args):
        episode_iter, epsilon, observation, do_buffer_update, do_explore = episode_args
        self.epsilon = epsilon
        with torch.no_grad():
            policy_output = self.apply_policy(
                observation, self.actor, epsilon, do_explore, False
            )
            action_indices = policy_output['indices']
            new_observation, user_feedback, update_info, _ = self.env.step(
                {'action': action_indices}
            )
            components = self._reward_components(
                observation, action_indices, user_feedback
            )
            objective_reward = components['total']
            click_reward = components['click']
            user_feedback['reward'] = objective_reward

            # Keep the shared evaluation metrics comparable with the other
            # single-response baselines.  The replay buffer still receives the
            # composite FairAgent objective used for learning.
            self.current_sum_reward += click_reward
            done_mask = user_feedback['done']
            if torch.any(done_mask):
                completed = self.current_sum_reward[done_mask]
                self.eval_history['avg_total_reward'].append(completed.mean().item())
                self.eval_history['max_total_reward'].append(completed.max().item())
                self.eval_history['min_total_reward'].append(completed.min().item())
                self.current_sum_reward[done_mask] = 0

            self.eval_history['avg_reward'].append(click_reward.mean().item())
            self.eval_history['reward_variance'].append(
                torch.var(click_reward).item()
            )
            for response_index, response in enumerate(self.env.response_types):
                self.eval_history[f'{response}_rate'].append(
                    user_feedback['immediate_response'][:, :, response_index].mean().item()
                )
            self._record_reward_components(components)

            if do_buffer_update:
                self.buffer.update(
                    observation,
                    policy_output,
                    user_feedback,
                    update_info['updated_observation'],
                )
        return new_observation

    def step_train(self):
        observation, policy_output, user_feedback, done_mask, next_observation = (
            self.buffer.sample(self.batch_size)
        )
        candidates = self.env.get_candidate_info(observation)
        current_q = self.actor.evaluate_actions(
            observation, policy_output['action'], candidates
        )

        with torch.no_grad():
            next_output = self.apply_policy(
                next_observation, self.actor_target, 0.0, False, True
            )
            next_q = next_output['preds'].sum(dim=1)
            reward = user_feedback['reward'].view(-1)
            target_q = reward + self.gamma * (~done_mask).float() * next_q

        # Equation (7).
        loss = F.mse_loss(current_q, target_q)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        self.fair_train_step += 1
        if self.fair_train_step % self.target_update_interval == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_target.eval()

        loss_dict = {
            'loss': float(loss.item()),
            'Q': float(current_q.mean().item()),
            'next_Q': float(next_q.mean().item()),
            'target_Q': float(target_q.mean().item()),
        }
        for key, value in loss_dict.items():
            self.training_history[key].append(value)
        return loss_dict

    def _prepare_test_step(self, step_index, total_steps, user_exit):
        if user_exit:
            dynamic_stage = self.actor.dynamic_stage_count
        else:
            dynamic_stage = min(
                self.actor.dynamic_stage_count,
                (
                    step_index * self.actor.dynamic_stage_count
                    // total_steps
                ) + 1,
            )
        self.actor.set_dynamic_stage(dynamic_stage)

    def _get_test_reward_and_metrics(
        self, observation, action_indices, user_feedback
    ):
        components = self._reward_components(
            observation, action_indices, user_feedback
        )
        stage = self.actor.current_dynamic_stage
        metric_sources = {
            'objective_reward': 'total',
            'accuracy_reward': 'accuracy',
            'fairness_reward': 'fairness',
            'new_item_reward': 'new_item',
            'new_item_coverage': 'new_item_coverage',
            'ad_reward': 'ad_reward',
            'slate_ad': 'slate_ad',
            'TGF': 'tgf',
            'UNF': 'unf',
        }
        metrics = {
            metric: components[source]
            for metric, source in metric_sources.items()
        }
        metrics.update({
            f'stage_{stage}_{source}': components[source]
            for source in metric_sources.values()
        })
        # ``avg_reward`` now has the same click-rate meaning as DDPG and the
        # other single-response baselines; ``objective_reward`` retains the
        # FairAgent-specific composite objective for diagnosis.
        return components['click'].detach(), metrics

    def test(self, test_env):
        """Use the shared repeated user-exit test and retain FairAgent metrics."""
        previous_dynamic_stage = self.actor.current_dynamic_stage
        try:
            return super().test(test_env)
        finally:
            self.actor.set_dynamic_stage(previous_dynamic_stage)

    def load(self):
        super().load()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()
