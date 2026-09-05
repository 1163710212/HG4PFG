#!/usr/bin/env bash
set -euo pipefail

THIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_SCRIPT_DIR}/_split_env.sh"

DATASET="Kuairand"
setup_split_environment "Kuairand" "Kuairand_Pure" "user_KRMBUserResponse_lr0.0001_reg0_nlayer2" "run_multibehavior.sh"

# environment args
ENV_CLASS='KREnvironment_WholeSession_GPU'
MAX_STEP="${MAX_STEP:-30}"
SLATE_SIZE=10
RHO=0.2
EPISODE_BATCH_SIZE=64
AD_TEMPER_PENALTY="${AD_TEMPER_PENALTY:-1.0}"

# policy args
POLICY_CLASS='OneStagePolicy_Sac'
HA_VAR=0.1
HA_CLIP=1.0
# if explore the effect action set --policy_do_effect_action_explore

# critic args
CRITIC_CLASS='TwinnedQCritic'


# buffer args
BUFFER_CLASS='SacBuffer'
BUFFER_SIZE=20000

# agent args
AGENT_CLASS='SAC'
GAMMA="${GAMMA:-0.99}"
REWARD_FUNC='get_immediate_reward'
N_ITER="${N_ITER:-30000}"
START_STEP="${START_STEP:-100}"
INITEP=0.1
ELBOW=0.1
EXPLORE_RATE=1.0
BS=64
SAC_TARGET_ENTROPY_RATIO="${SAC_TARGET_ENTROPY_RATIO:-0.8}"
SAC_GRAD_CLIP_NORM="${SAC_GRAD_CLIP_NORM:-10.0}"
CRITIC_LR_VALUE="${CRITIC_LR:-0.0003}"
# if want to explore in train set --do_explore_in_train
MEG=''


for HA_VAR in 0.1
do
    for REG in 0.00001
    do
        for INITEP in 0.01
        do
            for CRITIC_LR in "${CRITIC_LR_VALUE}"
            do
                for ACTOR_LR in 0.0001
                do
                    for SEED in "${SEED:-12}"
                    do
                    
                        file_key=${AGENT_CLASS}
                        
                        mkdir -p ${output_path}agents/${file_key}/
                        runtime_env_dir="${output_path}agents/${file_key}/environment_logs"
                        train_env_log="${runtime_env_dir}/train.model.log"
                        test_env_log="${runtime_env_dir}/test.model.log"
                        mkdir -p "${runtime_env_dir}"
                        normalize_environment_log "${TRAIN_ENV_LOG}" "${train_env_log}"
                        normalize_environment_log "${TEST_ENV_LOG}" "${test_env_log}"

                        "${PYTHON_BIN}" -u train_actor_critic.py \
                            --env_class ${ENV_CLASS}\
                            --policy_class ${POLICY_CLASS}\
                            --critic_class ${CRITIC_CLASS}\
                            --buffer_class ${BUFFER_CLASS}\
                            --agent_class ${AGENT_CLASS}\
                            --seed ${SEED}\
                            --dataset ${DATASET}\
                            --cuda "${CUDA_DEVICE}"\
                            --max_step_per_episode ${MAX_STEP}\
                            --initial_temper ${MAX_STEP}\
                            --ad_temper_penalty "${AD_TEMPER_PENALTY}"\
                            --uirm_log_path "${train_env_log}"\
                            --test_uirm_log_path "${test_env_log}"\
                            --dataset_dir "${DATASET_DIR}"\
                            --test_repeat "${TEST_REPEAT}"\
                            --test_num_users "${TEST_NUM_USERS}"\
                            --test_seed "${TEST_SEED}"\
                            --slate_size ${SLATE_SIZE}\
                            --episode_batch_size "${EPISODE_BATCH_SIZE}"\
                            --item_correlation ${RHO}\
                            --single_response\
                            --policy_action_hidden 256 64\
                            --state_user_latent_dim 16\
                            --state_item_latent_dim 16\
                            --state_transformer_enc_dim 32\
                            --state_transformer_n_head 4\
                            --state_transformer_d_forward 64\
                            --state_transformer_n_layer 3\
                            --state_dropout_rate 0.1\
                            --critic_hidden_dims 256 64\
                            --critic_dropout_rate 0.1\
                            --buffer_size ${BUFFER_SIZE}\
                            --gamma ${GAMMA}\
                            --reward_func ${REWARD_FUNC}\
                            --n_iter ${N_ITER}\
                            --train_every_n_step 1\
                            --start_policy_train_at_step ${START_STEP}\
                            --initial_epsilon ${INITEP}\
                            --final_epsilon ${INITEP}\
                            --elbow_epsilon ${ELBOW}\
                            --explore_rate ${EXPLORE_RATE}\
                            --check_episode 10\
                            --save_episode 200\
                            --save_path ${output_path}agents/${file_key}/model\
                            --actor_lr ${ACTOR_LR}\
                            --actor_decay ${REG}\
                            --batch_size ${BS}\
                            --critic_lr ${CRITIC_LR}\
                            --critic_decay ${REG}\
                            --target_mitigate_coef 0.01\
                            --sac_target_entropy_ratio "${SAC_TARGET_ENTROPY_RATIO}"\
                            --sac_grad_clip_norm "${SAC_GRAD_CLIP_NORM}"\
                            > ${output_path}agents/${file_key}/log
                    done
                done
            done
        done
    done
done
