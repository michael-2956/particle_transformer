#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-nnph4 kin --num-neurons-per-head 4

./scaling_experiments/train_and_test_with_settings.sh ParT-nnph64 kin --num-neurons-per-head 64

