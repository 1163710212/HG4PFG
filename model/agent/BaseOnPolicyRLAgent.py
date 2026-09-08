import time
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm

import utils
from model.reward import *
from model.agent.final_test_metrics import (
    add_exposure_metrics_to_report,
    append_exposure_metrics,
    get_exposure_metrics,
)
import json


class BaseOnPolicyRLAgent():
    """
    RL Agent controls the overall learning algorithm:
    - objective functions for the policies and critics
    - design of reward function
    - how many steps to train
    - how to do exploration
    - loading and saving of models

    Main interfaces:
    - train
    """

    @staticmethod
    def parse_model_args(parser):
        """
        args:
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
        # basic settings
        parser.add_argument('--gamma', type=float, default=0.95,
                            help='reward discount')
        parser.add_argument('--reward_func', type=str, default='get_retention_reward',
                            help='reward function name')
        parser.add_argument('--n_iter', type=int, nargs='+', default=[2000],
                            help='number of training iterations')
        parser.add_argument('--train_every_n_step', type=int, default=1,
                            help='number of training iterations')

        # exploration control
        parser.add_argument('--initial_epsilon', type=float, default=0.5,
                            help='probability for using uniform exploration')
        parser.add_argument('--final_epsilon', type=float, default=0.01,
                            help='probability for using uniform exploration')
        parser.add_argument('--elbow_epsilon', type=float, default=1.0,
                            help='probability for using uniform exploration')
        parser.add_argument('--explore_rate', type=float, default=1.0,
                            help='probability of engaging exploration')
        parser.add_argument('--do_explore_in_train', action='store_true',
                            help='probability of engaging exploration')

        # monitoring
        parser.add_argument('--check_episode', type=int, default=100,
                            help='number of iterations to check output and evaluate')
        parser.add_argument('--save_episode', type=int, default=1000,
                            help='number of iterations to save models')
        parser.add_argument('--save_path', type=str, required=True,
                            help='save path for networks')
        parser.add_argument('--test_n_step', type=int, default=100,
                            help='number of fixed-policy rollout steps in the final isolated test environment')
        parser.add_argument('--test_seed', type=int, default=2027,
                            help='fixed seed for the final isolated test rollout')
        parser.add_argument('--test_repeat', type=int, default=1,
                            help='number of independent final test repetitions')
        parser.add_argument('--test_num_users', type=int, default=0,
                            help='distinct users per repetition; 0 keeps the legacy fixed-step test')
        parser.add_argument('--verbose_rollout_debug', action='store_true',
                            help='print per-step fairness diagnostics during training rollouts')

        # learning
        parser.add_argument('--actor_lr', type=float, default=1e-4,
                            help='learning rate for actor')
        parser.add_argument('--actor_decay', type=float, default=1e-4,
                            help='regularization factor for actor learning')
        parser.add_argument('--batch_size', type=int, default=64,
                            help='training batch size')

        return parser

    def __init__(self, *input_args):
        args, env, actor, buffer = input_args

        self.device = args.device

        # hyperparameters
        self.gamma = args.gamma
        self.reward_func = eval(args.reward_func)
        self.n_iter = args.n_iter
        self.train_every_n_step = args.train_every_n_step

        self.initial_epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.elbow_epsilon = args.elbow_epsilon
        self.explore_rate = args.explore_rate
        self.do_explore_in_train = args.do_explore_in_train

        self.check_episode = args.check_episode
        self.save_episode = args.save_episode
        self.save_path = args.save_path
        self.test_n_step = args.test_n_step
        self.test_seed = args.test_seed
        self.test_repeat = args.test_repeat
        self.test_num_users = args.test_num_users
        self.verbose_rollout_debug = args.verbose_rollout_debug

        self.actor_lr = args.actor_lr
        self.actor_decay = args.actor_decay
        self.batch_size = args.batch_size

        # components
        self.env = env
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor.set_init_var()
        self.actor_target.set_init_var()
        self.buffer = buffer

        # controller
        self.exploration_scheduler = utils.LinearScheduler(int(sum(args.n_iter) * args.elbow_epsilon),
                                                           args.final_epsilon,
                                                           initial_p=args.initial_epsilon)
        self.registered_models = []

        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f" ")

    def train(self):
        self.setup_monitors()
        self.env.reset()
        self.buffer.reset(self.env, self.actor)
        self.actor_target.eval()

        if len(self.n_iter) > 2:
            self.load()

        t = time.time()
        start_time = t

        # training
        print("Training:")
        step_offset = sum(self.n_iter[:-1])
        do_buffer_update = True
        observation = deepcopy(self.env.current_observation)
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1] // 1)):
            do_explore = np.random.random() < self.explore_rate if self.explore_rate < 1 else True
            # online inference
            observation = self.run_episode_step(i, self.exploration_scheduler.value(i), observation,
                                                do_buffer_update, do_explore)
            # online training
            if (i + 1) % self.train_every_n_step == 0:
                self.step_train()
                self.buffer.clear()
            # log monitor records
            if i > 0 and i % self.check_episode == 0:
                t_prime = time.time()
                print(f"Episode step {i}, time diff {t_prime - t}, total time diff {t - start_time})")
                episode_report, train_report = self.get_report(smoothness=self.check_episode)
                episode_report = {k: float(v[0]) if isinstance(v, np.ndarray) else float(v)
                                  for k, v in episode_report.items()}
                train_report = {k: float(v[0]) if isinstance(v, np.ndarray) else float(v)
                                for k, v in train_report.items()}
                log_str = f"step: {i} @ online episode: {episode_report} @ training: {train_report}\n"
                with open(self.save_path + ".report", 'a') as outfile:
                    outfile.write(log_str)
                print(log_str)
                t = t_prime

            # save model and training info
            if i % self.save_episode == 0:
                self.save()

        # Always persist the final deployable actor/critic, even when the final
        # iteration is not an exact multiple of save_episode.
        self.save()
        self.action_after_train()

    def setup_monitors(self):
        self.training_history = {}
        self.eval_history = {'avg_reward': [],
                             'reward_variance': [],
                             'avg_total_reward': [0.],
                             'max_total_reward': [0.],
                             'min_total_reward': [0.],
                             'cof': [0],
                             }
        self.eval_history.update({f'{resp}_rate': [] for resp in self.env.response_types})
        self.current_sum_reward = torch.zeros(self.env.episode_batch_size).to(torch.float).to(self.device)
        self.current_step = torch.zeros(self.env.episode_batch_size).to(torch.long).to(self.device) 
        self.current_fair_weight = torch.zeros((self.env.episode_batch_size, 30)).to(torch.float).to(self.device)
        self.current_user_pop_prefer = torch.zeros((self.env.episode_batch_size, 30)).to(torch.float).to(self.device)
        self.user_pop_prefer = {}
        self.fair_weight = {}
        for i in range(40):
            self.user_pop_prefer[i] = []
            self.fair_weight[i] = []
        self.record_user_num = 0


    def action_after_train(self):
        self.env.stop()

    def get_report(self, smoothness=10):
        # Data recorded by the environment.
        episode_report = self.env.get_report(smoothness)
        # Data recorded by the agent.
        train_report = {k: np.mean(v[-smoothness:]) for k, v in self.training_history.items()}
        train_report.update({k: np.mean(v[-smoothness:]) for k, v in self.eval_history.items()})
        return episode_report, train_report

    def run_episode_step(self, *episode_args):
        """
        Run one step of user-env interaction
        @input:
        - episode_args: (episode_iter, epsilon, observation, do_buffer_update, do_explore)
        @process:
        - apply_policy: observation, candidate items --> policy_output
        - env.step(): policy_output['action'] --> user_feedback, updated_observation
        - reward_func(): user_feedback --> reward
        - buffer.update(observation, policy_output, user_feedback, updated_observation)
        @output:
        - next_observation
        """
        episode_iter, epsilon, observation, do_buffer_update, do_explore = episode_args
        self.epsilon = epsilon
        is_train = False
        with torch.no_grad():
            # generate action from policy
            # LERLC conditions its LLM category planner on the current session
            # step. Other policies ignore this extra observation field.
            observation['current_step'] = self.current_step
            policy_output = self.apply_policy(observation, self.actor_target, epsilon, do_explore, is_train)
            fairness_keys = {'user_pop_prefer', 'fair_weight', 'unpopular_item_ratio'}
            has_fairness_output = fairness_keys.issubset(policy_output)
            if has_fairness_output:
                user_pop_prefer = policy_output['user_pop_prefer']
                fair_weight = policy_output['fair_weight']
                unpopular_item_ratio = policy_output['unpopular_item_ratio']
                cof1 = np.corrcoef(user_pop_prefer.cpu().reshape(-1).numpy(), fair_weight.cpu().reshape(-1).numpy())
                cof2 = np.corrcoef(unpopular_item_ratio.cpu().reshape(-1).numpy(), fair_weight.cpu().reshape(-1).numpy())
                cof3 = np.corrcoef(unpopular_item_ratio.cpu().reshape(-1).numpy(), user_pop_prefer.cpu().reshape(-1).numpy())

            if has_fairness_output and episode_iter > 0 and self.verbose_rollout_debug:
                # print(unpopular_item_ratio)
                # print(fair_weight)
                print("-----------------------------------------------------")
                print(torch.mean(user_pop_prefer).cpu().reshape(-1).numpy(), 
                      torch.mean(unpopular_item_ratio).cpu().reshape(-1).numpy(), 
                      torch.mean(fair_weight).cpu().reshape(-1).numpy())
                print(cof1[0, 1], cof2[0, 1], cof3[0, 1])

            # apply action on environment
            # Note: action must be indices on env.candidate_iids
            action_dict = {'action': policy_output['indices']}
            new_observation, user_feedback, update_info, current_step = self.env.step(action_dict)
            self.current_step = current_step

            # calculate reward
            R = self.get_reward(user_feedback)
            user_feedback['reward'] = R
            self.current_sum_reward = self.current_sum_reward + R
            done_mask = user_feedback['done']

            # # Track user popularity preferences at different steps.
            if has_fairness_output and episode_iter > 0: #10000:
                current_step = current_step.cpu().numpy().reshape(-1)
                user_pop_prefer = user_pop_prefer.cpu().numpy().reshape(-1).tolist()
                fair_weight = fair_weight.cpu().numpy().reshape(-1).tolist()
                for i in range(len(current_step)):
                    self.user_pop_prefer[current_step[i]].append(user_pop_prefer[i])
                    self.fair_weight[current_step[i]].append(fair_weight[i])
                self.record_user_num += len(current_step)
                if self.record_user_num > 12800:
                    with open('./output/user_pop_prefer.txt', 'w') as file:
                        json.dump(self.user_pop_prefer, file, ensure_ascii=False)
                    with open('./output/fair_weight.txt', 'w') as file:
                        json.dump(self.fair_weight, file, ensure_ascii=False)
                    self.record_user_num = 0
                    for i in range(40):
                        self.user_pop_prefer[i] = []
                        self.fair_weight[i] = []
            
            # #
            # self.current_user_pop_prefer[:, self.current_step] = user_pop_prefer
            # self.current_fair_weight[:, self.current_step] = fair_weight
            # self.current_step += 1

            # Aggregate historical data for users who have left.
            if torch.sum(done_mask) > 0:
                self.eval_history['avg_total_reward'].append(self.current_sum_reward[done_mask].mean().item())
                self.eval_history['max_total_reward'].append(self.current_sum_reward[done_mask].max().item())
                self.eval_history['min_total_reward'].append(self.current_sum_reward[done_mask].min().item())
                self.current_sum_reward[done_mask] = 0

                if getattr(self, 'enable_reflection', False):
                    reflection_observation = deepcopy(new_observation)
                    self.reflect({
                        'history': reflection_observation['user_history']['history'][done_mask],
                        'single_step_reward': reflection_observation['user_history']['history_is_click'][done_mask],
                        'total_step': self.current_step[done_mask],
                    })

                # # 
                # temp_fair_weight = self.current_fair_weight[done_mask]
                # temp_user_pop_prefer = self.current_user_pop_prefer[done_mask]
                # temp_step = self.current_step[done_mask]
                # for i in torch.range(0, torch.sum(done_mask) - 1).long():
                #     #print(i, temp_step.shape, temp_user_pop_prefer.shape, torch.sum(done_mask))
                #     t_step = temp_step[i]
                #     t_fair_weight = temp_fair_weight[i, :t_step].cpu().reshape(-1).numpy()
                #     t_user_pop_prefer = temp_user_pop_prefer[i, :t_step].cpu().reshape(-1).numpy()
                # self.eval_history['cof'].append(np.corrcoef(t_fair_weight, t_user_pop_prefer)[0, 1])
                # self.current_step[done_mask] = 0

            # monitor update
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())

            for i, resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(user_feedback['immediate_response'][:, :, i].mean().item())
                # update replay buffer
            if do_buffer_update:
                if getattr(self, 'tracks_next_step', False):
                    self.buffer.update(
                        observation, policy_output, user_feedback,
                        update_info['updated_observation'], self.current_step + 1
                    )
                else:
                    self.buffer.update(observation, policy_output, user_feedback, update_info['updated_observation'])
        return new_observation

    def apply_policy(self, observation, actor, *policy_args):
        pass

    def reflect(self, feed_dict):
        """Optional terminal-session reflection hook used by LERLC."""
        return None

    def get_reward(self, user_feedback):
        user_feedback['immediate_response_weight'] = self.env.response_weights
        R = self.reward_func(user_feedback).detach()
        return R

    def step_train(self):
        """
        @process:
        """
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)

        loss_dict = self.get_loss(observation, policy_output, user_feedback, done_mask, next_observation)

        for k in loss_dict:
            if k in self.training_history:
                try:
                    self.training_history[k].append(loss_dict[k].item())
                except:
                    self.training_history[k].append(loss_dict[k])

    def get_loss(self, observation, policy_output, user_feedback, done_mask, next_observation):
        pass

    def _prepare_test_repeat(self, test_env):
        """Hook for policies that carry rollout state across environment steps."""

    def _run_fixed_step_test_repeat(self, test_env, repeat_seed):
        """Run the legacy fixed-horizon evaluation for compatibility."""
        self._prepare_test_repeat(test_env)
        observation = deepcopy(test_env.reset())
        current_step = torch.zeros(
            test_env.episode_batch_size, dtype=torch.long, device=self.device
        )
        running_returns = torch.zeros(
            test_env.episode_batch_size, device=self.device
        )
        completed_returns = []
        reward_samples = []
        response_samples = {response: [] for response in test_env.response_types}
        exposure_metric_samples = {}

        with torch.no_grad():
            for _ in range(self.test_n_step):
                observation['current_step'] = current_step
                policy_output = self.apply_policy(
                    observation, self.actor, 0.0, False, False, test_env
                )
                action_dict = {'action': policy_output['indices']}
                observation, user_feedback, _, current_step = test_env.step(
                    action_dict
                )

                user_feedback['immediate_response_weight'] = test_env.response_weights
                reward = self.reward_func(user_feedback).detach()
                running_returns += reward
                reward_samples.append(reward.detach().cpu().numpy())
                append_exposure_metrics(
                    exposure_metric_samples,
                    get_exposure_metrics(test_env, policy_output['indices']),
                )

                done_mask = user_feedback['done']
                if torch.any(done_mask):
                    completed_returns.extend(
                        running_returns[done_mask].detach().cpu().tolist()
                    )
                    running_returns[done_mask] = 0

                for response_idx, response in enumerate(test_env.response_types):
                    response_samples[response].append(
                        user_feedback['immediate_response'][:, :, response_idx]
                        .detach().cpu().numpy().reshape(-1)
                    )

        flat_rewards = np.concatenate(reward_samples)
        all_returns = completed_returns + running_returns.detach().cpu().tolist()
        test_report = {
            'seed': repeat_seed,
            'rollout_steps': self.test_n_step,
            'episode_batch_size': test_env.episode_batch_size,
            'completed_episodes': len(completed_returns),
            'avg_reward': float(np.mean(flat_rewards)),
            'reward_variance': float(np.var(flat_rewards)),
            'avg_total_reward': float(np.mean(all_returns)),
            'max_total_reward': float(np.max(all_returns)),
            'min_total_reward': float(np.min(all_returns)),
        }
        for response, values in response_samples.items():
            test_report[f'{response}_rate'] = float(
                np.mean(np.concatenate(values))
            )
        add_exposure_metrics_to_report(test_report, exposure_metric_samples)
        for metric, value in test_env.get_report(
            smoothness=self.test_n_step
        ).items():
            test_report[f'env_{metric}'] = float(value)
        return test_report

    def _run_user_exit_test_repeat(self, test_env, repeat_seed):
        """Evaluate distinct users until every selected user's first exit."""
        if not hasattr(test_env, 'reset_for_unique_users'):
            raise TypeError(
                f"{type(test_env).__name__} does not support unique-user testing"
            )

        self._prepare_test_repeat(test_env)
        observation = deepcopy(test_env.reset_for_unique_users(
            self.test_num_users, repeat_seed
        ))
        initial_user_ids = observation['user_profile']['user_id']
        unique_user_count = int(torch.unique(initial_user_ids).numel())
        if unique_user_count != self.test_num_users:
            raise RuntimeError(
                f"Expected {self.test_num_users} unique users, got "
                f"{unique_user_count}"
            )

        current_step = torch.zeros(
            self.test_num_users, dtype=torch.long, device=self.device
        )
        active_mask = torch.ones(
            self.test_num_users, dtype=torch.bool, device=self.device
        )
        running_returns = torch.zeros(self.test_num_users, device=self.device)
        episode_lengths = torch.zeros(
            self.test_num_users, dtype=torch.long, device=self.device
        )
        completed_returns = []
        reward_samples = []
        response_samples = {response: [] for response in test_env.response_types}
        exposure_metric_samples = {}
        rollout_steps = 0

        with torch.no_grad():
            while torch.any(active_mask):
                active_before_step = active_mask.clone()
                observation['current_step'] = current_step
                policy_output = self.apply_policy(
                    observation, self.actor, 0.0, False, False, test_env
                )
                action_dict = {'action': policy_output['indices']}
                observation, user_feedback, _, current_step = test_env.step(
                    action_dict
                )

                user_feedback['immediate_response_weight'] = test_env.response_weights
                reward = self.reward_func(user_feedback).detach()
                active_rewards = reward[active_before_step]
                running_returns[active_before_step] += active_rewards
                episode_lengths[active_before_step] += 1
                reward_samples.append(active_rewards.detach().cpu().numpy())
                append_exposure_metrics(
                    exposure_metric_samples,
                    get_exposure_metrics(
                        test_env,
                        policy_output['indices'],
                        active_before_step,
                    ),
                )

                for response_idx, response in enumerate(test_env.response_types):
                    response_samples[response].append(
                        user_feedback['immediate_response'][
                            active_before_step, :, response_idx
                        ].detach().cpu().numpy().reshape(-1)
                    )

                done_now = user_feedback['done'].bool() & active_before_step
                if torch.any(done_now):
                    completed_returns.extend(
                        running_returns[done_now].detach().cpu().tolist()
                    )
                active_mask = active_before_step & (~done_now)
                rollout_steps += 1

        if len(completed_returns) != self.test_num_users:
            raise RuntimeError(
                f"Expected {self.test_num_users} completed users, got "
                f"{len(completed_returns)}"
            )

        flat_rewards = np.concatenate(reward_samples)
        lengths = episode_lengths.detach().cpu().numpy()
        test_report = {
            'seed': repeat_seed,
            'rollout_steps': rollout_steps,
            'total_user_steps': int(lengths.sum()),
            'selected_users': self.test_num_users,
            'unique_users': unique_user_count,
            'completed_episodes': len(completed_returns),
            'avg_episode_length': float(np.mean(lengths)),
            'max_episode_length': int(np.max(lengths)),
            'min_episode_length': int(np.min(lengths)),
            'avg_reward': float(np.mean(flat_rewards)),
            'reward_variance': float(np.var(flat_rewards)),
            'avg_total_reward': float(np.mean(completed_returns)),
            'max_total_reward': float(np.max(completed_returns)),
            'min_total_reward': float(np.min(completed_returns)),
        }
        for response, values in response_samples.items():
            test_report[f'{response}_rate'] = float(
                np.mean(np.concatenate(values))
            )
        add_exposure_metrics_to_report(test_report, exposure_metric_samples)
        return test_report

    @staticmethod
    def _aggregate_test_repeats(repeat_reports):
        aggregate = {
            'repeat_count': len(repeat_reports),
            'seeds': [report['seed'] for report in repeat_reports],
            'completed_episodes': int(sum(
                report['completed_episodes'] for report in repeat_reports
            )),
            'rollout_steps_per_repeat': [
                report['rollout_steps'] for report in repeat_reports
            ],
            'repeat_reports': repeat_reports,
        }
        if 'selected_users' in repeat_reports[0]:
            aggregate.update({
                'users_per_repeat': repeat_reports[0]['selected_users'],
                'total_selected_users': int(sum(
                    report['selected_users'] for report in repeat_reports
                )),
                'total_user_steps': int(sum(
                    report['total_user_steps'] for report in repeat_reports
                )),
            })

        mean_metrics = [
            'avg_reward',
            'reward_variance',
            'avg_total_reward',
            'avg_episode_length',
        ]
        mean_metrics.extend(
            key for key in repeat_reports[0]
            if (
                key.endswith('_rate')
                or key.startswith('env_')
                or key in ('ad', 'coverage')
            )
        )
        for metric in dict.fromkeys(mean_metrics):
            if metric not in repeat_reports[0]:
                continue
            values = np.asarray(
                [report[metric] for report in repeat_reports], dtype=np.float64
            )
            aggregate[metric] = float(np.mean(values))
            aggregate[f'{metric}_std'] = float(np.std(values))

        aggregate['max_total_reward'] = float(max(
            report['max_total_reward'] for report in repeat_reports
        ))
        aggregate['min_total_reward'] = float(min(
            report['min_total_reward'] for report in repeat_reports
        ))
        if 'max_episode_length' in repeat_reports[0]:
            aggregate['max_episode_length'] = int(max(
                report['max_episode_length'] for report in repeat_reports
            ))
            aggregate['min_episode_length'] = int(min(
                report['min_episode_length'] for report in repeat_reports
            ))
        return aggregate

    def test(self, test_env):
        """Evaluate the saved policy in an isolated final test environment.

        The rollout is deterministic on the policy side (no Gaussian or
        multinomial exploration), never writes to the training buffer and never
        performs an optimizer step. Global RNG state is restored afterwards so
        this method is safe to reuse in controlled experiments.
        """
        if self.test_repeat <= 0:
            raise ValueError("test_repeat must be positive")
        if self.test_num_users < 0:
            raise ValueError("test_num_users cannot be negative")
        if self.test_num_users == 0 and self.test_n_step <= 0:
            raise ValueError("test_n_step must be positive")

        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        actor_was_training = self.actor.training
        previous_env = self.env

        try:
            self.actor.eval()
            self.env = test_env
            repeat_reports = []
            for repeat_index in range(self.test_repeat):
                repeat_seed = self.test_seed + repeat_index
                utils.set_random_seed(repeat_seed)
                if self.test_num_users > 0:
                    report = self._run_user_exit_test_repeat(
                        test_env, repeat_seed
                    )
                else:
                    report = self._run_fixed_step_test_repeat(
                        test_env, repeat_seed
                    )
                report['repeat'] = repeat_index + 1
                repeat_reports.append(report)
                if self.test_repeat > 1:
                    repeat_log = (
                        "final_test_env_repeat: "
                        + json.dumps(report, sort_keys=True)
                        + "\n"
                    )
                    with open(self.save_path + ".report", 'a') as outfile:
                        outfile.write(repeat_log)
                    print(repeat_log)

            test_report = (
                repeat_reports[0]
                if self.test_repeat == 1
                else self._aggregate_test_repeats(repeat_reports)
            )
            log_str = (
                "final_test_env: "
                + json.dumps(test_report, sort_keys=True)
                + "\n"
            )
            with open(self.save_path + ".report", 'a') as outfile:
                outfile.write(log_str)
            print(log_str)
            return test_report
        finally:
            self.env = previous_env
            self.actor.train(actor_was_training)
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

    def save(self):
        for model, opt, prefix in self.registered_models:
            torch.save(model.state_dict(), self.save_path + prefix)
            torch.save(opt.state_dict(), self.save_path + prefix + "_optimizer")

    def load(self):
        for model, opt, prefix in self.registered_models:
            model.load_state_dict(torch.load(self.save_path + prefix, map_location=self.device))
            opt.load_state_dict(torch.load(self.save_path + prefix + "_optimizer", map_location=self.device))
