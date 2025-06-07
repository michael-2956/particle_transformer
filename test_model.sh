#!/bin/bash

./scaling_experiments/test_with_settings.sh \
    ParT-esm0.5-samewts kin --embedding-scale-mult 0.5 \
    --network-option identical_attn_weights True \
    --network-option use_moe True
