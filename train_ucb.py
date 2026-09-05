"""Train context-free UCB and evaluate it on the isolated test simulator."""

import argparse
import copy
import gc

import torch

from model.agent.UCB import UCB
from train_linucb import (
    environment_class,
    get_environment_signature,
    validate_test_environment,
)
import utils


def main():
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument('--env_class', type=str, required=True)
    initial_parser.add_argument('--dataset', type=str, required=True)
    initial_args, _ = initial_parser.parse_known_args()
    env_class = environment_class(initial_args.env_class)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--env_class', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--test_uirm_log_path', type=str, default='')
    parser.add_argument('--ad_bound', type=float, default=0.1)
    parser = env_class.parse_model_args(parser)
    parser = UCB.parse_model_args(parser)
    args = parser.parse_args()

    if args.cuda >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
        args.device = f'cuda:{args.cuda}'
    else:
        args.device = 'cpu'
    utils.set_random_seed(args.seed)

    print('Loading train environment')
    train_env = env_class(args)
    train_signature = get_environment_signature(train_env)
    print('Setup UCB')
    agent = UCB(args, train_env)
    print(
        f'Context-free UCB(n_arms={agent.n_arms}, alpha={agent.alpha}, '
        f'slate_size={agent.slate_size})'
    )

    try:
        agent.train()
    except KeyboardInterrupt:
        agent.save()
        raise

    if not args.test_uirm_log_path:
        return

    agent.env = None
    train_env = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('Loading isolated test environment')
    test_args = copy.deepcopy(args)
    test_args.uirm_log_path = args.test_uirm_log_path
    if args.test_num_users > 0:
        test_args.episode_batch_size = args.test_num_users
    test_env = env_class(test_args)
    try:
        validate_test_environment(train_signature, test_env)
        agent.test(test_env)
    finally:
        test_env.stop()


if __name__ == '__main__':
    main()
