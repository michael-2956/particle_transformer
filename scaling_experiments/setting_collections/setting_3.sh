#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-nlcm0.9 kin --num-cls-layers-mult 0.9

./scaling_experiments/train_and_test_with_settings.sh ParT-esm4 kin --embedding-scale-mult 4

