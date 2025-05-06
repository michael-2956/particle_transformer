#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm0.5 kin --pair-embedding-scale-mult 0.5

./scaling_experiments/train_and_test_with_settings.sh ParT-nl3 kin --total-num-layers 3

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.7 kin --num-cls-layers-mult 0.7

./scaling_experiments/train_and_test_with_settings.sh ParT-nl16 kin --total-num-layers 16

./scaling_experiments/train_and_test_with_settings.sh ParT-nl20 kin --total-num-layers 20

