import copy
import math
import os

import torch
import torch.nn.functional as F

from model.agent.BaseRLAgent import BaseRLAgent


class SAC(BaseRLAgent):
    """Stable discrete SAC for slate recommendation.

    AD is not injected as a reward, loss, gradient, quota, or re-ranking rule.
    The policy only observes the simulator reward and transition/termination
    dynamics.  In this environment an imbalanced slate lowers user temper, so
    a stable long-horizon Q function can learn its future-session cost.
    """

    @staticmethod
    def parse_model_args(parser):
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument(
            '--critic_lr', type=float, default=3e-4,
            help='critic learning rate',
        )
        parser.add_argument(
            '--critic_decay', type=float, default=1e-5,
            help='critic weight decay',
        )
        parser.add_argument(
            '--target_mitigate_coef', type=float, default=0.01,
            help='Polyak target-network update coefficient',
        )
        parser.add_argument(
            '--sac_target_entropy_ratio', type=float, default=0.8,
            help='target entropy as a fraction of log(catalog size)',
        )
        parser.add_argument(
            '--sac_grad_clip_norm', type=float, default=10.0,
            help='maximum actor/critic gradient norm; zero disables clipping',
        )
        return parser

    def __init__(self, *input_args):
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)

        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef
        self.gamma_n = args.gamma
        self.sac_target_entropy_ratio = args.sac_target_entropy_ratio
        self.sac_grad_clip_norm = args.sac_grad_clip_norm
        self.sac_train_step = 0

        if not 0.0 < self.tau <= 1.0:
            raise ValueError('target_mitigate_coef must be in (0, 1]')
        if not 0.0 < self.sac_target_entropy_ratio <= 1.0:
            raise ValueError('sac_target_entropy_ratio must be in (0, 1]')
        if self.sac_grad_clip_norm < 0.0:
            raise ValueError('sac_grad_clip_norm must be non-negative')

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        self.log_alpha = torch.tensor(
            [-1.5], dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.alpha = self.log_alpha.exp().detach()
        self.maximum_entropy = (
            math.log(self.actor.action_dim) * self.sac_target_entropy_ratio
        )

        # The twin networks are one estimator.  Updating them atomically avoids
        # duplicate checkpoints and partial optimizer steps.
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            weight_decay=self.critic_decay,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.actor_lr
        )
        self.registered_models.append(
            (self.critic, self.critic_optimizer, '_critic')
        )

    def setup_monitors(self):
        super().setup_monitors()
        self.training_history.update({
            'actor_loss': [],
            'critic_loss': [],
            'Q1': [],
            'Q2': [],
            'next_Q': [],
            'next_V': [],
            'entropy': [],
            'entropy_loss': [],
            'alpha': [],
        })

    def _pointwise_transition_reward(self, policy_output, user_feedback):
        """Build one supervised TD target for every selected item.

        The previous implementation fitted only the mean Q of the K selected
        items.  Individual Q values could therefore grow in opposite
        directions without changing that mean, after which the actor exploited
        the unconstrained positive values.  Per-item response targets remove
        this null space while using exactly the original environment reward.
        """
        action = policy_output['action'].long()
        batch_size, slate_size = action.shape
        if getattr(self.reward_func, '__name__', '') != 'get_immediate_reward':
            return user_feedback['reward'].view(-1, 1).expand(-1, slate_size)

        response = user_feedback['immediate_response'].view(
            batch_size, slate_size, -1
        )
        weights = self.env.response_weights.view(1, 1, -1)
        return (response * weights).sum(dim=2)

    def _clip_gradients(self, parameters):
        if self.sac_grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, self.sac_grad_clip_norm)

    def _temperature_loss(self, entropies):
        # With gradient descent this raises alpha below the target entropy and
        # lowers alpha above it.  The original sign implemented the opposite.
        return (
            self.log_alpha * (entropies.detach() - self.maximum_entropy)
        ).mean()

    def step_train(self):
        observation, replay_output, user_feedback, done_mask, next_observation = (
            self.buffer.sample(self.batch_size)
        )
        action = replay_output['action'].long()
        self.sac_train_step += 1
        self.alpha = self.log_alpha.exp().detach()

        point_reward = self._pointwise_transition_reward(
            replay_output, user_feedback
        )
        with torch.no_grad():
            next_policy_output = self.apply_policy(
                next_observation, self.actor, 0.0, False, True
            )
            next_action_probs, next_log_action_probs = self.actor.evaluate(
                next_policy_output
            )
            next_q = self.apply_critic(
                next_observation, next_policy_output, self.critic_target
            )['q']
            next_v = (
                next_action_probs
                * (next_q - self.alpha * next_log_action_probs)
            ).sum(dim=1)
            target_q = point_reward + (
                self.gamma_n
                * (~done_mask).float().unsqueeze(1)
                * next_v.unsqueeze(1)
            )

        # Replay-buffer states become stale as the user encoder changes.
        # Re-encoding makes online inference and gradient updates consistent.
        current_policy_output = self.apply_policy(
            observation, self.actor, 0.0, False, True
        )
        critic_out = self.apply_critic(
            observation,
            {'state': current_policy_output['state'].detach()},
            self.critic,
        )
        current_q1 = critic_out['q1'].gather(1, action)
        current_q2 = critic_out['q2'].gather(1, action)
        q1_loss = F.smooth_l1_loss(current_q1, target_q)
        q2_loss = F.smooth_l1_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self._clip_gradients(self.critic.parameters())
        self.critic_optimizer.step()

        action_probs, log_action_probs = self.actor.evaluate(
            current_policy_output
        )
        with torch.no_grad():
            q = self.apply_critic(
                observation,
                {'state': current_policy_output['state'].detach()},
                self.critic,
            )['q']
        entropies = -(action_probs * log_action_probs).sum(dim=1)
        soft_q = (action_probs * q).sum(dim=1)
        policy_loss = -(self.alpha * entropies + soft_q).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self._clip_gradients(self.actor.parameters())
        self.actor_optimizer.step()

        entropy_loss = self._temperature_loss(entropies)
        self.alpha_optimizer.zero_grad()
        entropy_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=-10.0, max=2.0)
            self.alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            for parameter, target_parameter in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_parameter.mul_(1.0 - self.tau)
                target_parameter.add_(self.tau * parameter)

        loss_dict = {
            'actor_loss': policy_loss.item(),
            'critic_loss': critic_loss.item(),
            'Q1': current_q1.mean().item(),
            'Q2': current_q2.mean().item(),
            'next_Q': next_q.mean().item(),
            'next_V': next_v.mean().item(),
            'entropy': entropies.mean().item(),
            'entropy_loss': entropy_loss.item(),
            'alpha': self.alpha.item(),
        }
        for key, value in loss_dict.items():
            if key in self.training_history:
                self.training_history[key].append(float(value))
        return loss_dict

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

    @staticmethod
    def apply_critic(observation, policy_output, critic):
        return critic({'state': policy_output['state']})

    def action_after_train(self):
        # BaseRLAgent only checkpoints on periodic steps.  Persist the actual
        # final parameters before releasing the simulator.
        self.save()
        super().action_after_train()

    def save(self):
        super().save()
        torch.save({
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'sac_train_step': self.sac_train_step,
        }, self.save_path + '_temperature')

    def load(self):
        super().load()
        temperature_path = self.save_path + '_temperature'
        if os.path.exists(temperature_path):
            state = torch.load(temperature_path, map_location=self.device)
            with torch.no_grad():
                self.log_alpha.copy_(state['log_alpha'].to(self.device))
            self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
            self.sac_train_step = int(state.get('sac_train_step', 0))
        self.alpha = self.log_alpha.exp().detach()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)
