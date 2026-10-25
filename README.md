# TimeMoDE: Towards a Unified Generative Model for Scarce Time Series with Domain Experts

The repo is the official implementation for the paper: [https://arxiv.org/pdf/2606.15172](https://arxiv.org/pdf/2606.15172)

## Running the Code

The following instructions explain how to run the code. 

## Datasets

Preprocessed datasets are available for both **pretraining** and **fine-tuning**:

Download from: https://drive.google.com/file/d/1EHO1EXBJYg1ohFKxmJjV548qKtbSkVhF/view?usp=drive_link

Unzip the contents into the empty `./data` directory.

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

## Citation

If you find this repo useful, please cite our paper!

~~~bash
@article{yao2026towards,
  title={Towards a Unified Generative Model for Scarce Time Series with Domain Experts},
  author={Yao, Zihao and Zheng, Qi and Zuo, Jiankai and Zhang, Yaying},
  journal={arXiv preprint arXiv:2606.15172},
  year={2026}
}
~~~

## Code

Thanks for the open sources papers listed below which TimeMoDE is build on.

[https://github.com/azencot-group/ImagenFew](https://github.com/azencot-group/ImagenFew)

[https://github.com/kunncheng/Diff-MoE](https://github.com/kunncheng/Diff-MoE)

[https://github.com/syrGitHub/TFPS](https://github.com/syrGitHub/TFPS)
