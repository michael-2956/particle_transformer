#!/bin/bash

source ~/.py-env/bin/activate
mkdir runs/$1
mv $2 runs/$1
tensorboard --logdir runs/$1 "${@:3}"
