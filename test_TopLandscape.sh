#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_TopLandscape`
DATADIR=${DATADIR_TopLandscape}
[[ -z $DATADIR ]] && DATADIR='./datasets/TopLandscape'
# set a comment via `COMMENT`
suffix=${COMMENT}

# PN, PFN, PCNN, ParT
model=$1
extraopts=""
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-3"
elif [[ "$model" == "ParT-Small" ]]; then
    modelopts="networks/example_ParticleTransformerSmall.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-3"
elif [[ "$model" == "ParT-FocalLoss" ]]; then
    modelopts="networks/example_ParticleTransformer_AlteredLoss.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-4"
elif [[ "$model" == "ParT-FineTune" ]]; then
    modelopts="networks/example_ParticleTransformer_finetune.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-4"
    extraopts="--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler none --load-model-weights models/ParT_kin.pt"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    lr="1e-2"
elif [[ "$model" == "PN-FineTune" ]]; then
    modelopts="networks/example_ParticleNet_finetune.py"
    lr="1e-3"
    extraopts="--optimizer-option lr_mult (\"fc_out.*\",50) --lr-scheduler none --load-model-weights models/ParticleNet_kin.pt"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="kin"
if [[ "${FEATURE_TYPE}" != "kin" ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# evaluate on all 3 subsets
for part in train val test; do
    echo "Processing ${part} data..."
    weaver \
        --predict \
        --data-test "${DATADIR}/${part}_file.parquet" \
        --data-config data/TopLandscape/top_${FEATURE_TYPE}.yaml \
        --network-config $modelopts \
        --model-prefix trained_models/part_small_misclass_it2.pt \
        --num-workers 1 \
        --fetch-step 1 \
        --in-memory \
        --batch-size 512 \
        --predict-gpus 0 \
        --gpus 0 \
        --log logs/TopLandscape_${model}_{auto}${suffix}.log \
        --predict-output pred_${part}_kaggle.root \
        --tensorboard TopLandscape_${FEATURE_TYPE}_${model}${suffix} \
        ${extraopts} "${@:3}"
done
