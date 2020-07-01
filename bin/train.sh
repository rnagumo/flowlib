
# Run training

# Settings
MODEL=glow
CUDA=0
BATCH_SIZE=64
STEPS=5000
TEST_INTERVAL=500
SAVE_INTERVAL=500

# Log path
export LOGDIR=./logs/
export EXPERIMENT_NAME=${MODEL}

# Dataset path
export DATASET_DIR=./data/
export DATASET_NAME=cifar

# Config for training
export CONFIG_PATH=./examples/config.json

python3 ./examples/train.py --cuda ${CUDA} --model ${MODEL} \
    --max-steps ${STEPS} --batch-size ${BATCH_SIZE} \
    --test-interval ${TEST_INTERVAL} --save-interval ${SAVE_INTERVAL}
