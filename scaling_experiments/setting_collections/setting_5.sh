#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-esm0.25 kin --embedding-scale-mult 0.25

./scaling_experiments/train_and_test_with_settings.sh ParT-nl8 kin --total-num-layers 8

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.4 kin --num-cls-layers-mult 0.4

./scaling_experiments/train_and_test_with_settings.sh ParT-nl42 kin --total-num-layers 42

