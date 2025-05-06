#!/bin/bash

# train default model
./scaling_experiments/train_and_test_with_settings.sh ParT-default-soap kin

# skip 10 as the default setting
for nl in 1 2 3 4 6 8 12 16 20 24 30 36 42 50; do
    ./scaling_experiments/train_and_test_with_settings.sh ParT-nl${nl} kin --total-num-layers ${nl}
done