import time
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
import json

import utils
from model.reward import *
from model.agent.final_test_metrics import (
    add_exposure_metrics_to_report,
    append_exposure_metrics,
    get_exposure_metrics,
)

class BaseRLAgent():
    '''
    RL Agent controls the overall learning algorithm:
    - objective functions for the policies and critics
    - design of reward function
    - how many steps to train
    - how to do exploration
    - loading and saving of models
    
    Main interfaces:
    - train
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
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
        '''
        # basic settings
        parser.add_argument('--gamma', type=float, default=0.95, 
                            help='reward discount')
        parser.add_argument('--reward_func', type=str, default='get_retention_reward', 
                            help='reward function name')
        parser.add_argument('--n_iter', type=int, nargs='+', default=[2000], 
                            help='number of training iterations')
        parser.add_argument('--train_every_n_step', type=int, default=1, 
                            help='number of training iterations')
        parser.add_argument('--start_policy_train_at_step', type=int, default=1000,
                            help='start timestamp for buffer sampling')
        
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
                            help='number of fixed-policy rollout steps in the isolated test environment')
        parser.add_argument('--test_seed', type=int, default=2027,
                            help='fixed seed for the isolated test rollout')
        parser.add_argument('--test_repeat', type=int, default=1,
                            help='number of independent final test repetitions')
        parser.add_argument('--test_num_users', type=int, default=0,
                            help='distinct users per repetition; 0 keeps the legacy fixed-step test')
        
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
        self.start_policy_train_at_step = args.start_policy_train_at_step
        
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
        
        self.actor_lr = args.actor_lr
        self.actor_decay = args.actor_decay
        self.batch_size = args.batch_size
        
        # components
        self.env = env
        self.actor = actor
        self.buffer = buffer
        
        # controller
        # Keep one-step smoke runs valid: a fractional elbow can otherwise be
        # truncated to zero and make LinearScheduler divide by zero.
        scheduler_steps = max(1, int(sum(args.n_iter) * args.elbow_epsilon))
        self.exploration_scheduler = utils.LinearScheduler(scheduler_steps,
                                                           args.final_epsilon, 
                                                           initial_p=args.initial_epsilon)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        # register modules that will be saved
        self.registered_models = [(self.actor, self.actor_optimizer, '_actor')]
        
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f" ")
    
    def train(self):
        if len(self.n_iter) > 2:
            self.load()
        
        t = time.time()
        print("Run procedures before training")
        self.action_before_train()
        t = time.time()
        start_time = t
        
        # training
        print("Training:")
        step_offset = sum(self.n_iter[:-1])
        do_buffer_update = True
        observation = deepcopy(self.env.current_observation)
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1]//1)):
            do_explore = np.random.random() < self.explore_rate if self.explore_rate < 1 else True
            # online inference
            observation = self.run_episode_step(i, self.exploration_scheduler.value(i), observation, 
                                                do_buffer_update, do_explore)
            # online training
            if i % self.train_every_n_step == 0:
                self.step_train()
            # log monitor records
            if i > 0 and i % self.check_episode == 0:
                t_prime = time.time()
                print(f"Episode step {i}, time diff {t_prime - t}, total time diff {t - start_time})")
                episode_report, train_report = self.get_report(smoothness = self.check_episode)
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
               
        self.action_after_train()
        
    
    def action_before_train(self):
        '''
        Action before training:
        - env.reset()
        - buffer.reset()
        - set up training monitors
            - training_history
            - eval_history
        - run several episodes of random actions to build-up the initial buffer
        '''
        
        observation = self.env.reset()
        self.buffer.reset(self.env, self.actor)
        
        # training monitors
        self.setup_monitors()
        
        episode_iter = 0 # zero training iteration
        pre_epsilon = 1.0 # uniform random explore before training
        do_buffer_update = True
        prepare_step = 0
        
        for i in tqdm(range(self.start_policy_train_at_step)):
            do_explore = np.random.random() < self.explore_rate
            observation = self.run_episode_step(episode_iter, pre_epsilon, observation, 
                                                do_buffer_update, do_explore)
            prepare_step += 1
        print(f"Total {prepare_step} prepare steps")
        
    def setup_monitors(self):
        self.training_history = {'actor_loss': []}
        self.eval_history = {'avg_reward': [],
                             'reward_variance': [],
                             'avg_total_reward': [0.],
                             'max_total_reward': [0.],
                             'min_total_reward': [0.]}
        self.eval_history.update({f'{resp}_rate': [] for resp in self.env.response_types})
        self.current_sum_reward = torch.zeros(self.env.episode_batch_size).to(torch.float).to(self.device)
        self.user_pop_prefer = {}
        self.user_pop_ratio = {}
        self.fair_weight = {}
        for i in range(40):
            self.user_pop_prefer[i] = []
            self.user_pop_ratio[i] = []
            self.fair_weight[i] = []
        self.record_user_num = 0
        
    
    def action_after_train(self):
        self.env.stop()
        
    def get_report(self, smoothness = 10):
        episode_report = self.env.get_report(smoothness)
        train_report = {k: np.mean(v[-smoothness:]) for k,v in self.training_history.items()}
        train_report.update({k: np.mean(v[-smoothness:]) for k,v in self.eval_history.items()})
        return episode_report, train_report

    def run_episode_step(self, *episode_args):
        '''
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
        '''
        episode_iter, epsilon, observation, do_buffer_update, do_explore = episode_args
        self.epsilon = epsilon
        is_train = False
        with torch.no_grad():
            # generate action from policy
            policy_output = self.apply_policy(observation, self.actor, epsilon, do_explore, is_train)
            
            # apply action on environment
            # Note: action must be indices on env.candidate_iids
            action_dict = {'action': policy_output['indices']}
            new_observation, user_feedback, update_info, current_step = self.env.step(action_dict)
            
            # calculate reward
            R = self.get_reward(user_feedback)
            user_feedback['reward'] = R
            self.current_sum_reward = self.current_sum_reward + R
            done_mask = user_feedback['done']
            if torch.sum(done_mask) > 0:
                self.eval_history['avg_total_reward'].append(self.current_sum_reward[done_mask].mean().item())
                self.eval_history['max_total_reward'].append(self.current_sum_reward[done_mask].max().item())
                self.eval_history['min_total_reward'].append(self.current_sum_reward[done_mask].min().item())
                self.current_sum_reward[done_mask] = 0
            
            # monitor update
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())
            
            for i,resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(user_feedback['immediate_response'][:,:,i].mean().item())  
            # update replay buffer
            if do_buffer_update:
                self.buffer.update(observation, policy_output, user_feedback, update_info['updated_observation'])

            # Some fairness-aware policies expose this diagnostic. Standard
            # agents such as SAC do not, so monitoring must remain optional.
            user_pop_prefer = policy_output.get('user_pop_prefer')
            if user_pop_prefer is not None and episode_iter > 0: #10000:
                print("-----------------------------------------------------")
                print(torch.mean(user_pop_prefer).cpu().reshape(-1).numpy())
                current_step = current_step.cpu().numpy().reshape(-1)
                user_pop_prefer = user_pop_prefer.cpu().numpy().reshape(-1).tolist()
                for i in range(len(current_step)):
                    self.user_pop_prefer[current_step[i]].append(user_pop_prefer[i])
                self.record_user_num += len(current_step)
                if self.record_user_num > 12800:
                    with open('./output/user_pop_prefer.txt', 'w') as file:
                        json.dump(self.user_pop_prefer, file, ensure_ascii=False)
                    self.record_user_num = 0
                    for i in range(40):
                        self.user_pop_prefer[i] = []
        return new_observation
    
    def apply_policy(self, observation, actor, *input_args):
        '''
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
        '''
        epsilon = policy_args[0]
        do_explore = policy_args[1]
        is_train = policy_args[2]
        input_dict = {'observation': observation, 
                      'candidates': self.env.get_candidate_info(observation), 
                      'epsilon': epsilon, 
                      'do_explore': do_explore, 
                      'is_train': is_train, 
                      'batch_wise': False}
        out_dict = self.actor(input_dict)
        return out_dict
    
    def get_reward(self, user_feedback):
        user_feedback['immediate_response_weight'] = self.env.response_weights
        R = self.reward_func(user_feedback).detach()
        return R
    
    def step_train(self):
        '''
        @process:
        '''
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
        """Hook for policies that carry rollout state across test steps."""

    def _prepare_test_step(self, step_index, total_steps, user_exit):
        """Hook for policies with test-time state that changes by step."""

    def _get_test_reward_and_metrics(
        self, observation, action_indices, user_feedback
    ):
        """Return the test reward and optional per-user diagnostic tensors."""
        user_feedback['immediate_response_weight'] = self.env.response_weights
        reward = self.reward_func(user_feedback).detach()
        return reward, {}

    @staticmethod
    def _append_test_metrics(metric_samples, metrics, sample_mask=None):
        for key, value in metrics.items():
            if sample_mask is not None:
                value = value[sample_mask]
            metric_samples.setdefault(key, []).append(
                value.detach().cpu().numpy().reshape(-1)
            )

    @staticmethod
    def _add_test_metrics_to_report(report, metric_samples):
        for key, values in metric_samples.items():
            if values:
                report[key] = float(np.mean(np.concatenate(values)))

    def _run_fixed_step_test_repeat(self, test_env, repeat_seed):
        """Run the legacy fixed-horizon evaluation for compatibility."""
        self._prepare_test_repeat(test_env)
        observation = deepcopy(test_env.reset())
        running_returns = torch.zeros(
            test_env.episode_batch_size, device=self.device
        )
        completed_returns = []
        reward_samples = []
        response_samples = {response: [] for response in test_env.response_types}
        metric_samples = {}
        exposure_metric_samples = {}

        with torch.no_grad():
            for test_step in range(self.test_n_step):
                self._prepare_test_step(test_step, self.test_n_step, False)
                previous_observation = observation
                policy_output = self.apply_policy(
                    previous_observation, self.actor, 0.0, False, False
                )
                action_indices = policy_output['indices']
                observation, user_feedback, _, _ = test_env.step(
                    {'action': action_indices}
                )
                reward, metrics = self._get_test_reward_and_metrics(
                    previous_observation, action_indices, user_feedback
                )
                running_returns += reward
                reward_samples.append(reward.detach().cpu().numpy())
                self._append_test_metrics(metric_samples, metrics)
                append_exposure_metrics(
                    exposure_metric_samples,
                    get_exposure_metrics(test_env, action_indices),
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
        report = {
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
            report[f'{response}_rate'] = float(np.mean(np.concatenate(values)))
        self._add_test_metrics_to_report(report, metric_samples)
        add_exposure_metrics_to_report(report, exposure_metric_samples)
        for metric, value in test_env.get_report(
            smoothness=self.test_n_step
        ).items():
            report[f'env_{metric}'] = float(value)
        return report

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
        metric_samples = {}
        exposure_metric_samples = {}
        rollout_steps = 0

        with torch.no_grad():
            while torch.any(active_mask):
                active_before_step = active_mask.clone()
                self._prepare_test_step(rollout_steps, None, True)
                previous_observation = observation
                # org
                # policy_output = self.apply_policy(
                #     previous_observation, self.actor, 0.0, False, False
                # )
                policy_output = self.apply_policy(
                                    previous_observation, self.actor, 0.1, True, False
                                )
                action_indices = policy_output['indices']
                observation, user_feedback, _, _ = test_env.step(
                    {'action': action_indices}
                )

                reward, metrics = self._get_test_reward_and_metrics(
                    previous_observation, action_indices, user_feedback
                )
                active_rewards = reward[active_before_step]
                running_returns[active_before_step] += active_rewards
                episode_lengths[active_before_step] += 1
                reward_samples.append(active_rewards.detach().cpu().numpy())
                self._append_test_metrics(
                    metric_samples, metrics, active_before_step
                )
                append_exposure_metrics(
                    exposure_metric_samples,
                    get_exposure_metrics(
                        test_env, action_indices, active_before_step
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
        report = {
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
            report[f'{response}_rate'] = float(np.mean(np.concatenate(values)))
        self._add_test_metrics_to_report(report, metric_samples)
        add_exposure_metrics_to_report(report, exposure_metric_samples)
        return report

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
                key.endswith(('_rate', '_reward', '_coverage'))
                or key.startswith(('env_', 'stage_'))
                or key in ('TGF', 'UNF', 'ad', 'slate_ad', 'coverage')
            )
            and key not in ('max_total_reward', 'min_total_reward')
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
        """Evaluate the saved policy in the isolated final test environment."""
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
            model.load_state_dict(torch.load(self.save_path + prefix, map_location = self.device))
            opt.load_state_dict(torch.load(self.save_path + prefix + "_optimizer", map_location = self.device))
    
