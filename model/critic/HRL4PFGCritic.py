import torch.nn as nn

from model.components import DNN


class HRL4PFGCritic(nn.Module):
    """Independent high- and low-level value functions for HRL4PFG."""

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            '--critic_hidden_dims', type=int, nargs='+', default=[256, 64],
            help='hidden dimensions of both value networks',
        )
        parser.add_argument(
            '--critic_dropout_rate', type=float, default=0.1,
            help='dropout rate used by both value networks',
        )
        return parser

    def __init__(self, args, environment, policy):
        super().__init__()
        self.h_net = DNN(
            policy.state_dim,
            args.critic_hidden_dims,
            1,
            dropout_rate=args.critic_dropout_rate,
            do_batch_norm=True,
        )
        self.l_net = DNN(
            2 * policy.goal_dim,
            args.critic_hidden_dims,
            1,
            dropout_rate=args.critic_dropout_rate,
            do_batch_norm=True,
        )

    def high_value(self, high_state):
        return self.h_net(high_state).reshape(-1)

    def low_value(self, low_state):
        return self.l_net(low_state).reshape(-1)

    def forward(self, feed_dict):
        return {
            'h_v': self.high_value(feed_dict['h_state']),
            'l_v': self.low_value(feed_dict['l_state']),
        }

