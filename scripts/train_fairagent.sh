#!/usr/bin/env bash
set -euo pipefail

THIS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_SCRIPT_DIR}/_split_env.sh"

FAIR_ENVIRONMENT="${FAIR_ENVIRONMENT:-Kuairand}"
case "${FAIR_ENVIRONMENT}" in
    Kuairand)
        DATASET="Kuairand"
        # Reuse the dataset contract recorded by copied response-model logs.
        # An explicit DATASET_DIR from a sweep launcher still takes precedence.
        if [[ -z "${DATASET_DIR:-}" ]]; then
            recorded_env_log="${PROJECT_ROOT}/output/Kuairand_Pure/env/train/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log"
            if [[ -f "${recorded_env_log}" ]]; then
                recorded_dataset_dir="$(sed -n "2s/.*dataset_dir='\([^']*\)'.*/\1/p" "${recorded_env_log}")"
                if [[ -n "${recorded_dataset_dir}" && -d "${recorded_dataset_dir}" ]]; then
                    DATASET_DIR="${recorded_dataset_dir}"
                fi
            fi
        fi
        setup_split_environment "Kuairand" "Kuairand_Pure" \
            "user_KRMBUserResponse_lr0.0001_reg0_nlayer2" "run_multibehavior.sh"
        ;;
    KuaiRec)
        DATASET="KuaiRec"
        setup_split_environment "KuaiRec" "KuaiRec" \
            "user_KRMBUserResponse_lr0.0001_reg0_nlayer2" "run_multibehavior_rec.sh"
        ;;
    *)
        echo "FAIR_ENVIRONMENT must be Kuairand or KuaiRec" >&2
        exit 1
        ;;
esac

CUDA_DEVICE="${CUDA_DEVICE:-0}"

# Environment: use the same K=10 slate contract as the local baselines so
# click-rate and AD comparisons are on the same exposure granularity.
ENV_CLASS="KREnvironment_WholeSession_GPU"
MAX_STEP="${MAX_STEP:-30}"
SLATE_SIZE="${SLATE_SIZE:-10}"
RHO="${RHO:-0.2}"
EPISODE_BATCH_SIZE=64
AD_TEMPER_PENALTY="${AD_TEMPER_PENALTY:-1.0}"

# FairAgent uses DQN, not A2C.  Its policy retains the A2C actor parameter
# layout so embeddings and the ranking head can be inherited as in the paper.
POLICY_CLASS="FairAgentPolicy"
CRITIC_CLASS="VCritic"
BUFFER_CLASS="DqnBuffer"
AGENT_CLASS="FairAgent"

BACKBONE_ACTOR_PATH="${FAIR_BACKBONE_ACTOR_PATH:-${output_path}agents/A2C/model_actor}"
if [[ ! -f "${BACKBONE_ACTOR_PATH}" ]]; then
    echo "Missing FairAgent backbone actor: ${BACKBONE_ACTOR_PATH}" >&2
    echo "Train it first with: bash ${THIS_SCRIPT_DIR}/train_a2c.sh" >&2
    exit 1
fi

BUFFER_SIZE="${BUFFER_SIZE:-20000}"
CANDIDATE_SIZE="${CANDIDATE_SIZE:-1000}"
BACKBONE_RATIO="${BACKBONE_RATIO:-0.3}"
HISTORY_LENGTH="${HISTORY_LENGTH:-20}"

# Paper defaults/released configuration: alpha=2, beta=0.5, gamma=0.1.
FAIR_ALPHA="${FAIR_ALPHA:-2.0}"
FAIR_BETA="${FAIR_BETA:-0.5}"
FAIR_NEW_GAMMA="${FAIR_NEW_GAMMA:-0.1}"
TARGET_UPDATE_INTERVAL="${TARGET_UPDATE_INTERVAL:-5}"
# Learn popular/long-tail balance from the simulator's AD feedback.  This is a
# reward weight, not a fixed exposure quota; set it to 0 for the paper-only
# old/new objective.
FAIR_AD_WEIGHT="${FAIR_AD_WEIGHT:-0.5}"

# The paper denotes the future-reward discount by lambda.  This framework's
# shared option is named gamma, so RL_DISCOUNT is kept separate from the
# new-item reward gamma above.
RL_DISCOUNT="${RL_DISCOUNT:-0.9}"
N_ITER=30000
START_STEP="${START_STEP:-100}"
INITIAL_EPSILON="${INITIAL_EPSILON:-1.0}"
FINAL_EPSILON="${FINAL_EPSILON:-0.01}"
ELBOW_EPSILON="${ELBOW_EPSILON:-0.5}"
EXPLORE_RATE="${EXPLORE_RATE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACTOR_LR="${ACTOR_LR:-0.0001}" # 0.0001
ACTOR_DECAY="${ACTOR_DECAY:-0.00001}"
CRITIC_LR="${CRITIC_LR:-0.001}" #0.001
SEED="${SEED:-12}"

file_key="FairAgent_alpha${FAIR_ALPHA}_beta${FAIR_BETA}_gamma${FAIR_NEW_GAMMA}_adw${FAIR_AD_WEIGHT}_seed${SEED}"
run_dir="${output_path}agents/${file_key}"
mkdir -p "${run_dir}"

"${PYTHON_BIN}" train_actor_critic.py \
    --env_class "${ENV_CLASS}" \
    --policy_class "${POLICY_CLASS}" \
    --critic_class "${CRITIC_CLASS}" \
    --buffer_class "${BUFFER_CLASS}" \
    --agent_class "${AGENT_CLASS}" \
    --seed "${SEED}" \
    --dataset "${DATASET}" \
    --cuda "${CUDA_DEVICE}" \
    --max_step_per_episode "${MAX_STEP}" \
    --initial_temper "${MAX_STEP}" \
    --ad_temper_penalty "${AD_TEMPER_PENALTY}" \
    --uirm_log_path "${TRAIN_ENV_LOG}" \
    --test_uirm_log_path "${TEST_ENV_LOG}" \
    --dataset_dir "${DATASET_DIR}" \
    --test_repeat "${TEST_REPEAT}" \
    --test_num_users "${TEST_NUM_USERS}" \
    --test_seed "${TEST_SEED}" \
    --slate_size "${SLATE_SIZE}" \
    --episode_batch_size "${EPISODE_BATCH_SIZE}" \
    --item_correlation "${RHO}" \
    --single_response \
    --policy_action_hidden 256 64 \
    --policy_noise_var 0.1 \
    --policy_noise_clip 1.0 \
    --state_user_latent_dim 16 \
    --state_item_latent_dim 16 \
    --state_transformer_enc_dim 32 \
    --state_transformer_n_head 4 \
    --state_transformer_d_forward 64 \
    --state_transformer_n_layer 3 \
    --state_dropout_rate 0.1 \
    --fair_backbone_actor_path "${BACKBONE_ACTOR_PATH}" \
    --fair_candidate_size "${CANDIDATE_SIZE}" \
    --fair_backbone_ratio "${BACKBONE_RATIO}" \
    --fair_history_length "${HISTORY_LENGTH}" \
    --fair_positive_feedback is_click \
    --critic_hidden_dims 256 64 \
    --critic_dropout_rate 0.1 \
    --buffer_size "${BUFFER_SIZE}" \
    --gamma "${RL_DISCOUNT}" \
    --reward_func get_immediate_reward \
    --fair_alpha "${FAIR_ALPHA}" \
    --fair_beta "${FAIR_BETA}" \
    --fair_new_reward_gamma "${FAIR_NEW_GAMMA}" \
    --fair_target_update_interval "${TARGET_UPDATE_INTERVAL}" \
    --fair_ad_weight "${FAIR_AD_WEIGHT}" \
    --n_iter "${N_ITER}" \
    --train_every_n_step 1 \
    --start_policy_train_at_step "${START_STEP}" \
    --initial_epsilon "${INITIAL_EPSILON}" \
    --final_epsilon "${FINAL_EPSILON}" \
    --elbow_epsilon "${ELBOW_EPSILON}" \
    --explore_rate "${EXPLORE_RATE}" \
    --check_episode 10 \
    --save_episode 200 \
    --save_path "${run_dir}/model" \
    --actor_lr "${ACTOR_LR}" \
    --actor_decay "${ACTOR_DECAY}" \
    --batch_size "${BATCH_SIZE}" \
    --critic_lr "${CRITIC_LR}" \
    --critic_decay "${ACTOR_DECAY}" \
    > "${run_dir}/log" 2>&1
