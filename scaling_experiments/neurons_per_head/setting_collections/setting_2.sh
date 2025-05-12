#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-nnph2 kin --num-neurons-per-head 2

./scaling_experiments/train_and_test_with_settings.sh ParT-nnph32 kin --num-neurons-per-head 32

