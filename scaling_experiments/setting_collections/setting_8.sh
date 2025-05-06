#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm0.0625 kin --pair-embedding-scale-mult 0.0625

./scaling_experiments/train_and_test_with_settings.sh ParT-esm0.25 kin --embedding-scale-mult 0.25

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.8 kin --num-cls-layers-mult 0.8

./scaling_experiments/train_and_test_with_settings.sh ParT-nl16 kin --total-num-layers 16

./scaling_experiments/train_and_test_with_settings.sh ParT-nl20 kin --total-num-layers 20

