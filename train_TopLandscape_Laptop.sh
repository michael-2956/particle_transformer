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
    modelopts="networks/example_ParticleTransformer.py --optimizer-option weight_decay 0.01"
    lr="1e-3"
elif [[ "$model" == "ParT-Small" ]]; then
    modelopts="networks/example_ParticleTransformerSmall.py --optimizer-option weight_decay 0.01"
    lr="1e-3"
elif [[ "$model" == "ParT-AlteredLoss" ]]; then
    modelopts="networks/example_ParticleTransformer_AlteredLoss.py --optimizer-option weight_decay 0.01"
    lr="1e-4"
elif [[ "$model" == "ParT-Small-AlteredLoss" ]]; then
    modelopts="networks/example_ParticleTransformerSmall_AlteredLoss.py --optimizer-option weight_decay 0.01"
    lr="1e-3"
elif [[ "$model" == "ParT-FineTune" ]]; then
    modelopts="networks/example_ParticleTransformer_finetune.py --optimizer-option weight_decay 0.01"
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

# --data-test "${DATADIR}/test_file.parquet" \
# --no-mps \

weaver \
    --no-mps \
    --load-model-weights trained_models/parT_small_val7884.pt \
    --data-train "${DATADIR}/train_file.parquet" \
    --data-val "${DATADIR}/val_file.parquet" \
    --data-config data/TopLandscape/top_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/TopLandscape/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 1 --in-memory \
    --batch-size 128 --samples-per-epoch $((2400 * 512)) --samples-per-epoch-val $((800 * 512)) --num-epochs 20 --gpus "" \
    --start-lr $lr --optimizer ranger --log logs/TopLandscape_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard TopLandscape_${FEATURE_TYPE}_${model}${suffix} \
    ${extraopts} "${@:3}"
