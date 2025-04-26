#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_TopLandscape`
DATADIR=${DATADIR_TopLandscape}
[[ -z $DATADIR ]] && DATADIR='./datasets/TopLandscape'
# set a comment via `COMMENT`
suffix=${COMMENT}

lr="1e-3"
optimizer="ranger"
batch_size=512

# PN, PFN, PCNN, ParT
model=$1
extraopts=""
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "ParT-Trim16-SH" ]]; then
    modelopts="networks/example_ParticleTransformer_Trim16_SH.py --use-amp --optimizer-option weight_decay 0.01"
    lr="3e-3"
    optimizer="soap"
    batch_size=2048
elif [[ "$model" == "ParT-MultiplePairEmbeds" ]]; then
    modelopts="networks/ParticleTransformerMultiplePairEmbeds.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "InteractionTransformer" ]]; then
    modelopts="networks/InteractionTransformer.py --use-amp --optimizer-option weight_decay 0.01"
    lr="3e-3"
    optimizer="soap"
    batch_size=512
elif [[ "$model" == "ParT-Inverter" ]]; then
    modelopts="networks/ParticleTransformerWithInverter.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "ParT-Long" ]]; then
    modelopts="networks/example_ParticleTransformerLong.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "ParT-Wide" ]]; then
    modelopts="networks/example_ParticleTransformerWide.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "ParT-Big" ]]; then
    modelopts="networks/example_ParticleTransformerBig.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "ParT-Small" ]]; then
    modelopts="networks/example_ParticleTransformerSmall.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "ParT-Tiny" ]]; then
    modelopts="networks/ParticleTransformerTiny.py --use-amp --optimizer-option weight_decay 0.01"
    lr="3e-3"
    optimizer="soap"
    batch_size=2048
elif [[ "$model" == "ParT-Tiny-MultiplePairEmbeds" ]]; then
    modelopts="networks/ParticleTransformerTinyMultiplePairEmbeds.py --use-amp --optimizer-option weight_decay 0.01"
    lr="3e-3"
    optimizer="soap"
    batch_size=512  # 2048 gives out of memory
elif [[ "$model" == "ParT-Tiny-NoDrop" ]]; then
    modelopts="networks/ParticleTransformerTinyNoDropout.py --use-amp --optimizer-option weight_decay 0.01"
    lr="3e-3"
    optimizer="soap"
    batch_size=2048
elif [[ "$model" == "ParT-Pico-NoDrop" ]]; then
    modelopts="networks/ParticleTransformerPicoNoDropout.py --use-amp --optimizer-option weight_decay 0.01"
    lr="3e-3"
    optimizer="soap"
    batch_size=2048
elif [[ "$model" == "ParT-Pico-NoDrop-NoTrim" ]]; then
    modelopts="networks/ParticleTransformerPicoNoDropoutNoTrim.py --use-amp --optimizer-option weight_decay 0.01"
    lr="3e-3"
    optimizer="soap"
    batch_size=2048
elif [[ "$model" == "ParT-Nano-NoDrop-NoTrim" ]]; then
    modelopts="networks/ParticleTransformerNanoNoDropoutNoTrim.py --use-amp --optimizer-option weight_decay 0.01"
    lr="3e-3"
    optimizer="soap"
    batch_size=2048
elif [[ "$model" == "ParT-Nano-NoDrop-Trim16-Shuffle-noU-M128" ]]; then
    modelopts="networks/ParticleTransformer_Nano_NoDropout_trim16_shuffle_noU_Mult128.py --use-amp --optimizer-option weight_decay 0.01"
    lr="3e-3"
    optimizer="soap"
    batch_size=8196
elif [[ "$model" == "ParT-AlteredLoss" ]]; then
    modelopts="networks/example_ParticleTransformer_AlteredLoss.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "ParT-Long-AlteredLoss" ]]; then
    modelopts="networks/example_ParticleTransformerLong_AlteredLoss.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "ParT-Wide-AlteredLoss" ]]; then
    modelopts="networks/example_ParticleTransformerLong_AlteredLoss.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "ParT-Big-AlteredLoss" ]]; then
    modelopts="networks/example_ParticleTransformerLong_AlteredLoss.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ "$model" == "ParT-Small-AlteredLoss" ]]; then
    modelopts="networks/example_ParticleTransformerSmall_AlteredLoss.py --use-amp --optimizer-option weight_decay 0.01"
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
if [[ "${FEATURE_TYPE}" != "kin" && "${FEATURE_TYPE}" != "kin_aug" ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!  Allowed: kin | kin_aug"
    exit 1
fi

# (!) PASS THIS OPTION IN KAGGLE:
# --model-prefix trained_models/model.pt

preds_location="kaggle"
# preds_location="local_cpu"

# evaluate on all 3 subsets
for part in train val test; do
    echo "Processing ${part} data..."
    weaver \
        --predict \
        --data-test "${DATADIR}/${part}_file.parquet" \
        --data-config data/TopLandscape/top_${FEATURE_TYPE}.yaml \
        --network-config $modelopts \
        --num-workers 1 \
        --fetch-step 1 \
        --in-memory \
        --batch-size $batch_size \
        --predict-gpus 0 \
        --gpus 0 \
        --log logs/TopLandscape_${model}_{auto}${suffix}.log \
        --predict-output pred_${part}_${preds_location}.root \
        --tensorboard TopLandscape_${FEATURE_TYPE}_${model}${suffix} \
        ${extraopts} "${@:3}"
done
