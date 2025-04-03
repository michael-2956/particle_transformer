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

for part in train val test; do
    echo "Processing ${part} data..."
    weaver \
        --predict \
        --data-test "${DATADIR}/${part}_file.parquet" \
        --no-mps \
        --data-config data/TopLandscape/top_${FEATURE_TYPE}.yaml --network-config $modelopts \
        --model-prefix trained_models/local_small_on_ds_with_partmistakes_inverted_7eps.pt \
        --num-workers 1 --fetch-step 1 --in-memory \
        --batch-size 128 --predict-gpus "" --gpus "" \
        --log logs/TopLandscape_${model}_{auto}${suffix}.log \
        --predict-output pred_${part}_local_cpu.root \
        --tensorboard TopLandscape_${FEATURE_TYPE}_${model}${suffix} \
        ${extraopts} "${@:3}"
done
