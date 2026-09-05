#!/usr/bin/env bash

# Shared train/test-environment contract for RL experiment scripts.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif [[ -x "/home/choxia/anaconda3/envs/hdcrec/bin/python3.8" ]]; then
        PYTHON_BIN="/home/choxia/anaconda3/envs/hdcrec/bin/python3.8"
    else
        PYTHON_BIN="python"
    fi
fi
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TEST_N_STEP="${TEST_N_STEP:-100}"
TEST_SEED="${TEST_SEED:-2027}"
TEST_REPEAT="${TEST_REPEAT:-3}"
TEST_NUM_USERS="${TEST_NUM_USERS:-100}"

# Copied response-model logs can retain absolute paths to a sibling checkout.
# Normalize a run-local copy so dataset and checkpoint paths resolve inside the
# checkout that owns the RL run, without modifying the trained artifacts.
normalize_environment_log() {
    local source_log="$1"
    local target_log="$2"
    sed -E \
        -e "s|/home/choxia/Data/HER4IF-main/codex(-v[0-9]+)?/|${PROJECT_ROOT}/|g" \
        -e "s|/data/choxia/HER4IF-main/codex(-v[0-9]+)?/|${PROJECT_ROOT}/|g" \
        "${source_log}" > "${target_log}"
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    echo "Activate the project environment or set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi

setup_split_environment() {
    local dataset_name="$1"
    local output_name="$2"
    local environment_model_key="$3"
    local environment_setup_script="$4"

    DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/dataset/${dataset_name}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/${output_name}}"
    ENV_MODEL_KEY="${environment_model_key}"
    TRAIN_ENV_LOG="${OUTPUT_DIR}/env/train/log/${ENV_MODEL_KEY}.model.log"
    TEST_ENV_LOG="${OUTPUT_DIR}/env/test/log/${ENV_MODEL_KEY}.model.log"
    TRAIN_ENV_CHECKPOINT="${OUTPUT_DIR}/env/train/${ENV_MODEL_KEY}.model.checkpoint"
    TEST_ENV_CHECKPOINT="${OUTPUT_DIR}/env/test/${ENV_MODEL_KEY}.model.checkpoint"
    TRAIN_ENV_COMPLETE="${OUTPUT_DIR}/env/train/${ENV_MODEL_KEY}.model.complete"
    TEST_ENV_COMPLETE="${OUTPUT_DIR}/env/test/${ENV_MODEL_KEY}.model.complete"
    TRAIN_ENV_INCOMPLETE="${OUTPUT_DIR}/env/train/${ENV_MODEL_KEY}.model.incomplete"
    TEST_ENV_INCOMPLETE="${OUTPUT_DIR}/env/test/${ENV_MODEL_KEY}.model.incomplete"

    local required_file
    for required_file in \
        "${TRAIN_ENV_LOG}" \
        "${TEST_ENV_LOG}" \
        "${TRAIN_ENV_CHECKPOINT}" \
        "${TEST_ENV_CHECKPOINT}"
    do
        if [[ ! -f "${required_file}" ]]; then
            echo "Missing trained environment artifact: ${required_file}" >&2
            echo "Run: bash ${SCRIPT_DIR}/${environment_setup_script}" >&2
            exit 1
        fi
    done

    # New environment-training scripts publish explicit lifecycle markers.
    # Enforce them whenever this marker protocol is present, while keeping
    # older, already-trained datasets backward compatible.
    if [[ -e "${TRAIN_ENV_COMPLETE}" || -e "${TEST_ENV_COMPLETE}" || \
          -e "${TRAIN_ENV_INCOMPLETE}" || -e "${TEST_ENV_INCOMPLETE}" ]]
    then
        if [[ -e "${TRAIN_ENV_INCOMPLETE}" || -e "${TEST_ENV_INCOMPLETE}" ]]; then
            echo "Environment training is incomplete; rerun ${SCRIPT_DIR}/${environment_setup_script}" >&2
            exit 1
        fi
        for required_marker in "${TRAIN_ENV_COMPLETE}" "${TEST_ENV_COMPLETE}"
        do
            if [[ ! -f "${required_marker}" ]]; then
                echo "Missing environment completion marker: ${required_marker}" >&2
                echo "Run: bash ${SCRIPT_DIR}/${environment_setup_script}" >&2
                exit 1
            fi
        done
    fi

    output_path="${OUTPUT_DIR}/"
    log_name="${ENV_MODEL_KEY}"
    mkdir -p "${OUTPUT_DIR}/agents"
}
