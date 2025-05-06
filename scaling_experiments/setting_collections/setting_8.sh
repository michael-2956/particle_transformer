#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm0.0625 kin --pair-embedding-scale-mult 0.0625

./scaling_experiments/train_and_test_with_settings.sh ParT-nl2 kin --total-num-layers 2

# train default model
./scaling_experiments/train_and_test_with_settings.sh ParT-default-soap kin

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.8 kin --num-cls-layers-mult 0.8

./scaling_experiments/train_and_test_with_settings.sh ParT-nl30 kin --total-num-layers 30

