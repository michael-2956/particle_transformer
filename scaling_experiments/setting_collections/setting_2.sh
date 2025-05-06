#!/bin/bash

./scaling_experiments/train_and_test_with_settings.sh ParT-esm0.03125 kin --embedding-scale-mult 0.03125

./scaling_experiments/train_and_test_with_settings.sh ParT-esm4 kin --embedding-scale-mult 4

