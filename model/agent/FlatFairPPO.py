from model.agent.PPO import PPO


class FlatFairPPO(PPO):
    """Flat PPO trained with accuracy plus distributional fairness reward."""

    @staticmethod
    def parse_model_args(parser):
        parser = PPO.parse_model_args(parser)
        parser.add_argument(
            '--lambda_fairness', type=float, default=0.1,
            help=(
                'weight of normalized catalogue-entropy fairness in the '
                'flat PPO rollout reward'
            ),
        )
        return parser

    def __init__(self, args, environment, actor, critic, buffer):
        if args.lambda_fairness < 0.0:
            raise ValueError('lambda_fairness must be non-negative')
        super().__init__(args, environment, actor, critic, buffer)
        self.lambda_fairness = args.lambda_fairness
        self._rollout_fairness_reward = None

    @staticmethod
    def shape_flat_reward(
        accuracy_reward, fairness_reward, lambda_fairness
    ):
        return accuracy_reward + lambda_fairness * fairness_reward

    def setup_monitors(self):
        super().setup_monitors()
        self.eval_history.update({
            'accuracy_reward': [],
            'fairness_reward': [],
            'objective_reward': [],
        })

    def apply_policy(self, observation, actor, *policy_args):
        policy_output = super().apply_policy(
            observation, actor, *policy_args
        )
        if 'fairness_reward' not in policy_output:
            raise KeyError(
                'FlatFairPPO requires a policy output named fairness_reward'
            )
        # PPO calls get_reward immediately after each rollout action.  Cache
        # only a detached scalar per user; the policy gradient still follows
        # PPO's likelihood-ratio estimator through the shaped return.
        self._rollout_fairness_reward = policy_output[
            'fairness_reward'
        ].detach()
        return policy_output

    def get_reward(self, user_feedback):
        accuracy_reward = super().get_reward(user_feedback)
        if self._rollout_fairness_reward is None:
            raise RuntimeError(
                'fairness reward is unavailable before policy application'
            )
        fairness_reward = self._rollout_fairness_reward.to(
            device=accuracy_reward.device,
            dtype=accuracy_reward.dtype,
        )
        if fairness_reward.shape != accuracy_reward.shape:
            raise ValueError(
                'accuracy and fairness rewards must have the same shape'
            )
        objective_reward = self.shape_flat_reward(
            accuracy_reward, fairness_reward, self.lambda_fairness
        )
        if hasattr(self, 'eval_history'):
            self.eval_history['accuracy_reward'].append(
                accuracy_reward.mean().item()
            )
            self.eval_history['fairness_reward'].append(
                fairness_reward.mean().item()
            )
            self.eval_history['objective_reward'].append(
                objective_reward.mean().item()
            )
        return objective_reward.detach()
