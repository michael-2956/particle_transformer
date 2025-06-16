#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass-100M'

# set a comment via `COMMENT`
suffix=${COMMENT}

optimizer="soap"
batch_size=512
lr="1e-3"

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=50
# epochs=20
samples_per_epoch=$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=$((10000 * 128))
# samples_per_epoch=$((2000 * 512 / $NGPUS))
# samples_per_epoch_val=$((2500 * 512))
dataopts="--num-workers 2 --fetch-step 0.01"

model=$1
shift 1
if [[ $model == ParT* ]]; then
  modelopts="networks/scalable_ParticleTransformer.py --use-amp"
elif [[ $model == InT* ]]; then
  modelopts="networks/scalable_InteractionTransformer.py --use-amp"
else
  echo "Invalid model $model!"
  exit 1
fi

# "kin", "kinpid", "full"
FEATURE_TYPE=$1
shift 1
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"
if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# default settings
nl=10
nlcm=0.2
esm=1
pesm=1
nnph=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --total-num-layers)
      nl="$2"; shift 2;;
    --num-cls-layers-mult)
      nlcm="$2"; shift 2;;
    --embedding-scale-mult)
      esm="$2"; shift 2;;
    --pair-embedding-scale-mult)
      pesm="$2"; shift 2;;
    --num-neurons-per-head)
      nnph="$2"; shift 2;;
    *)
      break
      ;;
  esac
done

# currently only Pythia
SAMPLE_TYPE=Pythia

$CMD \
    --data-train \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/train_100M/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/train_100M/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/train_100M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/train_100M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/train_100M/ZJetsToNuNu_*.root" \
    --data-val "${DATADIR}/${SAMPLE_TYPE}/val_5M/*.root" \
    --data-test \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/test_20M/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/test_20M/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/test_20M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/test_20M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/test_20M/ZJetsToNuNu_*.root" \
    --data-config data/JetClass/JetClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/JetClass/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts \
    --samples-per-epoch ${samples_per_epoch} \
    --samples-per-epoch-val ${samples_per_epoch_val} \
    --num-epochs $epochs \
    --gpus 0 \
    --predict-gpus 0 \
    --optimizer $optimizer \
    --batch-size $batch_size \
    --start-lr $lr \
    --log logs/JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}_{auto}${suffix}.log \
    --predict-output pred.root \
    --tensorboard JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}${suffix} \
    --network-option total_num_layers ${nl} \
    --network-option num_cls_layers_mult ${nlcm} \
    --network-option embedding_scale_mult ${esm} \
    --network-option pair_embedding_scale_mult ${pesm} \
    --network-option num_neurons_per_head ${nnph} \
    "$@"

mkdir -p tested_models
latest=$(find training/JetClass/Pythia/full/${model} -maxdepth 1 -mindepth 1 -type d \
         | sort \
         | tail -n1)
cp "${latest}/net_best_epoch_state.pt"             tested_models/${model}_best.pt
cp "${latest}/net_epoch-$((epochs - 1))_state.pt"  tested_models/${model}_last.pt
