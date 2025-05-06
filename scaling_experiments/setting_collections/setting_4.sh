#!/bin/bash

# skip 1 as the default setting
for pesm in 0.03125 0.0625 0.125 0.25 0.5 2 4; do
    ./scaling_experiments/train_and_test_with_settings.sh ParT-pesm${pesm} kin --pair-embedding-scale-mult ${pesm}
done
