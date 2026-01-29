#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_ckpt=''

for subset_n in 10 25 50
do
  for dataset in "Mujoco" "ETTh2" "ETTm1" "ETTm2" "AAPL" "ILI" "SaugeenRiverFlow" "ECG200" "StarLightCurves" "AirQuality"
  do
    echo "Running dataset=$dataset with subset_n=$subset_n"
    python run.py \
      --subset_n $subset_n \
      --model_ckpt $model_ckpt \
      --config configs/finetune_DiTMoE/$dataset.yaml
  done
done