#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm1 kin --num-cls-layers-mult 1

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm4 kin --pair-embedding-scale-mult 4

