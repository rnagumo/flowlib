
# Run training
# $ bash bin/train.sh <seed> <conditional>

# Kwargs
SEED=${1:-0}
CONDITIONAL=${2:-0}

# Settings
MODEL=glow
CUDA=0
BATCH_SIZE=32
STEPS=10000
TEST_INTERVAL=1000
SAVE_INTERVAL=1000

# Log path
export LOGDIR=./logs/
export EXPERIMENT_NAME=${MODEL}

# Dataset path
export DATASET_DIR=./data/
export DATASET_NAME=cifar

# Config for training
export CONFIG_PATH=./examples/config.json

if [[ CONDITIONAL -eq 0 ]]; then
    python3 ./examples/train.py --cuda ${CUDA} --model ${MODEL} \
        --seed ${SEED} --max-steps ${STEPS} --batch-size ${BATCH_SIZE} \
        --test-interval ${TEST_INTERVAL} --save-interval ${SAVE_INTERVAL}
else
    # Conditional
    python3 ./examples/train.py --cuda ${CUDA} --model ${MODEL} \
        --seed ${SEED} --max-steps ${STEPS} --batch-size ${BATCH_SIZE} \
        --test-interval ${TEST_INTERVAL} --save-interval ${SAVE_INTERVAL} \
        --y-conditional
fi
