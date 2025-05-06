#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-nl1 kin --total-num-layers 1

./scaling_experiments/train_and_test_with_settings.sh ParT-nl4 kin --total-num-layers 4

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.1 kin --num-cls-layers-mult 0.1

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.6 kin --num-cls-layers-mult 0.6

./scaling_experiments/train_and_test_with_settings.sh ParT-nl24 kin --total-num-layers 24

