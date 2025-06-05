#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_TopLandscape`
DATADIR=${DATADIR_TopLandscape}
[[ -z $DATADIR ]] && DATADIR='./datasets/TopLandscape'
# set a comment via `COMMENT`
suffix=${COMMENT}

optimizer="soap"
batch_size=512
lr="3e-3"

model=$1
shift 1
extraopts=""

if [[ $model == ParT* ]]; then
  modelopts="networks/scalable_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"
elif [[ $model == InT* ]]; then
  modelopts="networks/scalable_InteractionTransformer.py --use-amp --optimizer-option weight_decay 0.01"
else
  echo "Invalid model $model!"
  exit 1
fi

# "kin"
FEATURE_TYPE=$1
shift 1
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="kin"  # kin by default
if [[ "${FEATURE_TYPE}" != "kin" && "${FEATURE_TYPE}" != "kin_aug" ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!  Allowed: kin | kin_aug"
    exit 1
fi

preds_location="cern_cluster"

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

weaver \
    --gpus 0 \
    --data-train "${DATADIR}/train_file.parquet" \
    --data-val "${DATADIR}/val_file.parquet" \
    --data-config data/TopLandscape/top_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/TopLandscape/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 1 --in-memory \
    --batch-size $batch_size \
    --samples-per-epoch $((2400 * 512)) \
    --samples-per-epoch-val $((800 * 512)) \
    --num-epochs 20 \
    --start-lr $lr \
    --optimizer $optimizer \
    --log logs/TopLandscape_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard TopLandscape_${FEATURE_TYPE}_${model}${suffix} \
    --network-option total_num_layers ${nl} \
    --network-option num_cls_layers_mult ${nlcm} \
    --network-option embedding_scale_mult ${esm} \
    --network-option pair_embedding_scale_mult ${pesm} \
    --network-option num_neurons_per_head ${nnph} \
    ${extraopts} "$@"

mkdir -p tested_models
latest=$(find training/TopLandscape/${model} -maxdepth 1 -mindepth 1 -type d \
         | sort \
         | tail -n1)
cp "${latest}/net_best_epoch_state.pt" tested_models/${model}_best.pt
cp "${latest}/net_epoch-19_state.pt"   tested_models/${model}_last.pt

for wt_path in ${model}_best ${model}_last; do
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
           --log logs/TopLandscape_${wt_path}_{auto}${suffix}.log \
           --predict-output pred_${wt_path}_${part}_${preds_location}.root \
           --tensorboard TopLandscape_${FEATURE_TYPE}_${wt_path}${suffix} \
           --model-prefix tested_models/${wt_path}.pt \
           --network-option total_num_layers ${nl} \
           --network-option num_cls_layers_mult ${nlcm} \
           --network-option embedding_scale_mult ${esm} \
           --network-option pair_embedding_scale_mult ${pesm} \
           --network-option num_neurons_per_head ${nnph} \
           ${extraopts} "$@"
   done
done
