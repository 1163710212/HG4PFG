"""Train disjoint LinUCB and evaluate it on the isolated test simulator."""

import argparse
import copy
import gc
import importlib

import torch

from model.agent.LinUCB import LinUCB
import utils


def environment_class(name):
    module = importlib.import_module('env.' + name)
    try:
        return getattr(module, name)
    except AttributeError as error:
        raise ValueError(f'Unknown environment class: {name}') from error


def get_environment_signature(environment):
    stats = environment.reader.get_statistics()
    return {
        'environment_split': stats.get('environment_split'),
        'n_user': stats['n_user'],
        'n_item': stats['n_item'],
        'max_seq_len': stats['max_seq_len'],
        'user_feature_dims': stats['user_feature_dims'],
        'item_feature_dims': stats['item_feature_dims'],
        'feedback_type': stats['feedback_type'],
        'slate_size': environment.slate_size,
        'candidate_iids': environment.candidate_iids.detach().cpu().clone(),
    }


def validate_test_environment(train_signature, test_environment):
    test_signature = get_environment_signature(test_environment)
    if train_signature['environment_split'] != 'train':
        raise ValueError(
            "Training checkpoint must use environment_split='train', got "
            f"{train_signature['environment_split']!r}"
        )
    if test_signature['environment_split'] != 'test':
        raise ValueError(
            "Test checkpoint must use environment_split='test', got "
            f"{test_signature['environment_split']!r}"
        )

    mismatches = []
    for key, train_value in train_signature.items():
        if key in ('environment_split', 'candidate_iids'):
            continue
        if train_value != test_signature[key]:
            mismatches.append(key)
    if not torch.equal(
        train_signature['candidate_iids'], test_signature['candidate_iids']
    ):
        mismatches.append('candidate_iids')
    if mismatches:
        raise ValueError(
            'Train/test environment spaces are incompatible: '
            + ', '.join(mismatches)
        )


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
    parser = LinUCB.parse_model_args(parser)
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
    print('Setup LinUCB')
    agent = LinUCB(args, train_env)
    print(
        f'Disjoint LinUCB(n_arms={agent.n_arms}, '
        f'context_dim={agent.context_dim}, alpha={agent.alpha}, '
        f'ridge={agent.ridge}, slate_size={agent.slate_size})'
    )

    try:
        agent.train()
    except KeyboardInterrupt:
        agent.save()
        raise

    if not args.test_uirm_log_path:
        return

    # KuaiRec has a large reader.  Do not retain train and test dataframes at
    # the same time.
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
