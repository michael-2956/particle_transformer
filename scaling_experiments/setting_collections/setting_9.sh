#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-esm0.0625 kin --embedding-scale-mult 0.0625

./scaling_experiments/train_and_test_with_settings.sh ParT-esm0.125 kin --embedding-scale-mult 0.125

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm0.5 kin --pair-embedding-scale-mult 0.5

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.5 kin --num-cls-layers-mult 0.5

./scaling_experiments/train_and_test_with_settings.sh ParT-nl12 kin --total-num-layers 12

./scaling_experiments/train_and_test_with_settings.sh ParT-esm2 kin --embedding-scale-mult 2

