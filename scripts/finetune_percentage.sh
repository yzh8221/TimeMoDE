export CUDA_VISIBLE_DEVICES=0

model_ckpt=''

for subset_p in 0.05 0.1 0.15
do
  for dataset in "Mujoco" "ETTh2" "ETTm1" "ETTm2" "AAPL" "ILI" "SaugeenRiverFlow" "ECG200" "StarLightCurves" "AirQuality"
  do
    echo "Running dataset=$dataset with subset_p=$subset_p"
    python run.py \
      --subset_p $subset_p \
      --model_ckpt $model_ckpt \
      --config configs/finetune/$dataset.yaml
  done
done
