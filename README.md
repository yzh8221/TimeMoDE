# TimeMoDE: Towards a Unified Generative Model for Scarce Time Series with Domain Experts

## Running the Code

The following instructions explain how to run the code. 

### Environment & Libraries

The full libraries list is provided as `requirements.txt`. Please create a virtual environment by running

~~~bash
conda create --name TimeMoDE python=3.10 pip
conda activate TimeMoDE
pip install -r requirements.txt
~~~

### Pretraining

For pre-training, you can reproduce the experimental results by runing

```bash
python run.py --no_test_model --seq_len <sequence_length> --config <config_file>
# or
./scripts/pretrain.sh
```

### Fine-tuning

Please use the saved checkpoints of pre-trained model for fine-tuning by running

Percentage-based:
```bash
python run.py --subset_p <percent> --model_ckpt <ckpt_path> --config <config_file>
```

Count-based:
```bash
python run.py --subset_n <count> --model_ckpt <ckpt_path> --config <config_file>
```

Scripts for all datasets:
```bash
./scripts/finetune_percentage.sh
./scripts/finetune_count.sh
```