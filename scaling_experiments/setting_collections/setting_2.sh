#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-nl2 kin --total-num-layers 2

./scaling_experiments/train_and_test_with_settings.sh ParT-esm0.5 kin --embedding-scale-mult 0.5

./scaling_experiments/train_and_test_with_settings.sh ParT-nl42 kin --total-num-layers 42

