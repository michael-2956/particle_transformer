#!/bin/bash

# skip 0.2 as the default setting
for nlcm in 0.1 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
    ./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm${nlcm} kin --num-cls-layers-mult ${nlcm}
done