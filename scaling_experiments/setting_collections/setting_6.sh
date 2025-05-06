#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-esm0.125 kin --embedding-scale-mult 0.125

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.3 kin --num-cls-layers-mult 0.3

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm1 kin --num-cls-layers-mult 1

./scaling_experiments/train_and_test_with_settings.sh ParT-esm2 kin --embedding-scale-mult 2

