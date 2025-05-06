#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm0.125 kin --pair-embedding-scale-mult 0.125

./scaling_experiments/train_and_test_with_settings.sh ParT-nl8 kin --total-num-layers 8

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.3 kin --num-cls-layers-mult 0.3

./scaling_experiments/train_and_test_with_settings.sh ParT-nl30 kin --total-num-layers 30

