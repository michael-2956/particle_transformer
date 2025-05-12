#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-nnph1 kin --num-neurons-per-head 1

./scaling_experiments/train_and_test_with_settings.sh ParT-nnph64 kin --num-neurons-per-head 64

