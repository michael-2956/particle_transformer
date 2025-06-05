#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh \
    ParT-samewts-WLD-W-nl30-nlcm0.06-esm0.5-nnph16 kin \
    --total-num-layers 30 \
    --num-cls-layers-mult 0.06 \
    --embedding-scale-mult 0.5 \
    --network-option weighted_decode_every_layer True \
    --network-option identical_attn_weights True \
    --network-option use_moe False
