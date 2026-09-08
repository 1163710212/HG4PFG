from tqdm import tqdm
from time import time
import copy
import gc
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os

from model.agent import *
from model.policy import *
from model.critic import *
from model.buffer import *
from env import KREnvironment_WholeSession_GPU, KREnvironment_WholeSession_GPUx

import utils


# Set CUDA_LAUNCH_BLOCKING to 1.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
            f"Training checkpoint must use environment_split='train', got "
            f"{train_signature['environment_split']!r}"
        )
    if test_signature['environment_split'] != 'test':
        raise ValueError(
            f"Test checkpoint must use environment_split='test', got "
            f"{test_signature['environment_split']!r}"
        )

    tensor_keys = {'candidate_iids'}
    mismatches = []
    for key in train_signature:
        if key in ('environment_split', *tensor_keys):
            continue
        if train_signature[key] != test_signature[key]:
            mismatches.append(key)
    if not torch.equal(train_signature['candidate_iids'], test_signature['candidate_iids']):
        mismatches.append('candidate_iids')
    if mismatches:
        raise ValueError(
            "Train/test environment spaces are incompatible: " + ", ".join(mismatches)
        )

if __name__ == '__main__':
    
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--env_class', type=str, required=True, help='Environment class.')
    init_parser.add_argument('--policy_class', type=str, required=True, help='Policy class')
    init_parser.add_argument('--critic_class', type=str, required=True, help='Critic class')
    init_parser.add_argument('--agent_class', type=str, required=True, help='Learning agent class')
    init_parser.add_argument('--buffer_class', type=str, required=True, help='Buffer class.')
    init_parser.add_argument('--dataset', type=str, required=True, help='Dataset.')
    init_parser.add_argument('--aug_weight', type=float, default=0.1, required=False, help='Aug weight.')
    init_parser.add_argument('--ad_bound', type=float, default=0.1, required=False, help='Ad bound.')


    initial_args, _ = init_parser.parse_known_args()
    print(f"mushxx{initial_args}")
    envClass = eval('{0}.{0}'.format(initial_args.env_class))
    policyClass = eval('{0}.{0}'.format(initial_args.policy_class))
    criticClass = eval('{0}.{0}'.format(initial_args.critic_class))
    agentClass = eval('{0}.{0}'.format(initial_args.agent_class))
    bufferClass = eval('{0}.{0}'.format(initial_args.buffer_class))
    
    # experimental control args
    # Several launchers pass --dataset after --dataset_dir.  This parser does
    # not own --dataset (the initial parser above does), so argparse's default
    # abbreviation logic would otherwise reinterpret --dataset as
    # --dataset_dir and overwrite the real directory with values such as
    # "Kuairand".
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--cuda', type=int, default=-1, help='cuda device number; set to -1 (default) if using cpu')
    parser.add_argument('--test_uirm_log_path', type=str, default='',
                        help='user-response model log for the isolated final test environment')
    
    # customized args
    parser = envClass.parse_model_args(parser)
    parser = policyClass.parse_model_args(parser)
    parser = criticClass.parse_model_args(parser)
    parser = agentClass.parse_model_args(parser)
    parser = bufferClass.parse_model_args(parser)
    args, _ = parser.parse_known_args()
    args.dataset = initial_args.dataset
    args.aug_weight = initial_args.aug_weight
    args.ad_bound = initial_args.ad_bound
    
    if args.cuda >= 0 and torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    args.device = device
    utils.set_random_seed(args.seed)
    
    # Environment
    print("Loading train environment")
    env = envClass(args)
    train_environment_signature = get_environment_signature(env)
    
    # Policy, Critic, Buffer, Agent
    print("Setup policy:")
    policy = policyClass(args, env)
    policy.to(device)
    print(policy)
    print("Setup critic:")
    if initial_args.agent_class == 'TD3':
        critic1 = criticClass(args, env, policy)
        critic1.to(device)
        critic2 = criticClass(args, env, policy)
        critic2.to(device)
        critic = [critic1, critic2]
    else:
        critic = criticClass(args, env, policy)
        critic.to(device)
    print(critic)
    print("Setup buffer:")
    buffer = bufferClass(args, env, policy, critic)
    print(buffer)
    print("Setup agent:")
    agent = agentClass(args, env, policy, critic, buffer)
    print(agent)
    
    # online training
    try:
        print(args)
        agent.train()
    except KeyboardInterrupt:
        print("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            print(os.linesep + '-' * 20 + ' END: ' + utils.get_local_time() + ' ' + '-' * 20)
            exit(1)

    if args.test_uirm_log_path:
        # KuaiRec contains more than 12 million rows. Release the train reader
        # and simulator before constructing the test environment so both full
        # dataframes and response models never need to coexist in memory.
        agent.env = None
        agent.buffer = None
        env.stop()
        env = None
        buffer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Loading isolated test environment")
        test_args = copy.deepcopy(args)
        test_args.uirm_log_path = args.test_uirm_log_path
        if getattr(args, 'test_num_users', 0) > 0:
            test_args.episode_batch_size = args.test_num_users
        test_env = envClass(test_args)
        try:
            validate_test_environment(train_environment_signature, test_env)
            agent.test(test_env)
        finally:
            test_env.stop()
