#!/usr/bin/env bash

CUDA=0,1
PORT=1235
COMMENT="deeplabv3p-resnet50-tversky"

CUDA_VISIBLE_DEVICES=$CUDA accelerate launch --config accelerate/multi-gpu.json --main_process_port $PORT run.py train \
    --data.path=$DATA_PATH \
    --model.decoder=deeplabv3p \
    --model.encoder=resnet50 \
    --trainer.batch-size=8 \
    --loss.target=combo \
    --trainer.max-epochs=100 \
    --trainer.no-amp \
    --optimizer.encoder-lr=1e-4 \
    --optimizer.decoder-lr=1e-3 \
    --scheduler.target=poly \
    --data.mask-body-ratio=0.0 \
    --data.no-include-dem \
    --model.multibranch \
    --data.in-channels=2 \
    --data.weighted-sampling \
    --visualize \
    --num-samples=4 \
    --comment $COMMENT
