#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm0.03125 kin --pair-embedding-scale-mult 0.03125

./scaling_experiments/train_and_test_with_settings.sh ParT-nl1 kin --total-num-layers 1

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.1 kin --num-cls-layers-mult 0.1

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.9 kin --num-cls-layers-mult 0.9

./scaling_experiments/train_and_test_with_settings.sh ParT-nl36 kin --total-num-layers 36

