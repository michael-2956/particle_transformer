#!/bin/bash

# skip 1 as the default setting
for esm in 0.03125 0.0625 0.125 0.25 0.5 2 4; do
    ./scaling_experiments/train_and_test_with_settings.sh ParT-esm${esm} kin --embedding-scale-mult ${esm}
done
