#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-esm0.03125 kin --embedding-scale-mult 0.03125

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm0.03125 kin --pair-embedding-scale-mult 0.03125

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm0.25 kin --pair-embedding-scale-mult 0.25

./scaling_experiments/train_and_test_with_settings.sh ParT-nl6 kin --total-num-layers 6

# train default model
./scaling_experiments/train_and_test_with_settings.sh ParT-default-soap kin

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.4 kin --num-cls-layers-mult 0.4

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm2 kin --pair-embedding-scale-mult 2

