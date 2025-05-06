#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm0.125 kin --pair-embedding-scale-mult 0.125

./scaling_experiments/train_and_test_with_settings.sh ParT-esm0.5 kin --embedding-scale-mult 0.5

./scaling_experiments/train_and_test_with_settings.sh ParT-pesm2 kin --pair-embedding-scale-mult 2

