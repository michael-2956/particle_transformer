#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-esm0.0625 kin --embedding-scale-mult 0.0625

./scaling_experiments/train_and_test_with_settings.sh ParT-nl6 kin --total-num-layers 6

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.5 kin --num-cls-layers-mult 0.5

./scaling_experiments/train_and_test_with_settings.sh ParT-nl50 kin --total-num-layers 50

