#!/usr/bin/env bash
# return to the root directory

CUDA=2,3,4,5
PORT=1235
COMMENT="whatever is relevant"

CUDA_VISIBLE_DEVICES=$CUDA accelerate launch --config accelerate/multi-gpu.json --main_process_port $PORT run.py train \
--data-root /mnt/userdata/montello_data/shub/imgs_zoom11/ready-to-train \
--model.encoder=tresnet_m \
--trainer.batch-size=8 \
--trainer.patience=30 \
--optimizer.lr=1e-3 \
--scheduler.target=cosine \
--comment "$COMMENT"
