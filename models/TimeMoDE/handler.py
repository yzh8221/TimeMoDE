from .TimeMoDE import TimeMoDE
from ..generative_handler import generativeHandler

import os
import torch
import logging
import torch.nn as nn
import numpy as np

dataset_list = [
            "stock",
            "energy",
            "ETTh1",
            "Exchange",
            "MSL",
            "PSM",
            "SMAP",
            "SMD",
            "SelfRegulationSCP2",
            "UWaveGestureLibrary",
            "ECG5000",
            "NonInvasiveFetalECGThorax1",
            "Blink",
            "ElectricDevices",
            "Trace",
            "FordB",
            "EMOPain",
            "Chinatown",
            "SharePriceIncrease",
        # FINE TUNE DATASETS BELOW
            "sine",
            "mujoco",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "Weather",
            "ILI",
            "SaugeenRiverFlow",
            "ECG200",
            "SelfRegulationSCP1",
            "StarLightCurves",
            "AirQuality",
            "AAPL",
        ]

class Handler(generativeHandler):

    def __init__(self, args, rank=None):
        super().__init__(args, rank)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.step = 0

    def build_model(self):
        self.model = TimeMoDE(self.args, self.args.device)
        if self.args.finetune:
            self._load_model(self.args.model_ckpt, self.args.device)
            if self.args.ema:
                self.model.setup_finetune(self.args)
        else:
            if self.args.model_ckpt is not None:
                self._load_model(self.args.model_ckpt, self.args.device)
        return self.model
    
    def train_iter(self, train_dataloader, logger, metadatas):
        MSE_avg = []
        for _, data in enumerate(train_dataloader, 1):

            x = data[0].to(self.device).float()
            class_indices = data[1].to(self.args.device)
            labels = nn.functional.one_hot(class_indices, num_classes = self.args.n_classes)

            self.optimizer.zero_grad()
            varn = metadatas[dataset_list[class_indices[0]]]['channels']
            loss = self.model(x, labels=labels, varn=varn)

            if isinstance(loss, tuple):
                loss, to_log = loss
                for key, value in to_log.items():
                    logger.log(f'train/{key}', value, self.epoch)
            else:
                logger.log(f'train/loss', loss.item(), self.epoch)

            MSE_avg.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self._model.on_train_batch_end()
            # self.step += 1

        print('epoch_MSE_avg:', np.mean(MSE_avg))

    def sample(self, train_set, n_samples, class_label, class_metadata):

        sample_set = []
        for _, data in enumerate(train_set, 1):
            sample = data[0].to(self.device).float()
            sample_set.append(sample)
        sample_set = torch.cat(sample_set, dim=0)
        print('few shot shape:', sample_set.shape)

        generated_set = []
        sample_id_start = 0
        with self._model.ema_scope():
            for sample_size in [min(self.args.batch_size//2, n_samples - i) for i in range(0, n_samples, self.args.batch_size//2)]:
                random_indices = torch.arange(sample_id_start, sample_id_start+sample_size)%sample_set.shape[0]
                sample_id_start = sample_id_start+sample_size
                random_samples = sample_set[random_indices]
                class_labels = torch.full((sample_size,), class_label, device=self.args.device)
                oh_class_labels = torch.nn.functional.one_hot(class_labels, num_classes=self.args.n_classes)
                x_sampled = self.model.generate_mts(random_samples, sample_size, labels=oh_class_labels, data_shape=(self.args.in_len, self.args.input_channels))
                x_ts = x_sampled[:,:,:class_metadata['channels']]
                generated_set.append(x_ts)
                
        return torch.concat(generated_set, dim=0)
    
    def save_model(self, ckpt_dir):
        state = {'model': self._model.state_dict()}
        if self.args.ema is not None:
            state['ema_model'] = self._model.model_ema.state_dict()
        torch.save(state, ckpt_dir)

    def _load_model(self, ckpt_dir, device):
        if not os.path.exists(ckpt_dir):
            os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
            logging.warning(f"No checkpoint found at {ckpt_dir}. "
                            f"Returned the same state as input")
        else:
            loaded_state = torch.load(ckpt_dir, map_location=device)
            self.model.load_state_dict(loaded_state['model'], strict=False)
            if 'ema_model' in loaded_state and self.args.ema is not None:
                self.model.model_ema.load_state_dict(loaded_state['ema_model'], strict=False)
            logging.info(f'Successfully loaded previous state')