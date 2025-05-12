#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-nnph1 kin --num-neurons-per-head 1 --gpus 0,1,2,3 --predict-gpus 0,1,2,3

