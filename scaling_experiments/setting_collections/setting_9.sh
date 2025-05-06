#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm0.25 kin --pair-embedding-scale-mult 0.25

./scaling_experiments/train_and_test_with_settings.sh ParT-nl4 kin --total-num-layers 4

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.6 kin --num-cls-layers-mult 0.6

./scaling_experiments/train_and_test_with_settings.sh ParT-nl12 kin --total-num-layers 12

./scaling_experiments/train_and_test_with_settings.sh ParT-nl24 kin --total-num-layers 24

