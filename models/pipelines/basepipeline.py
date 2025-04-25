#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:basepipeline.py
# author:xm
# datetime:2024/4/25 12:30
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import os

import torch
import torch.nn as nn
# import module your need
from accelerate import Accelerator
from torchmetrics import Accuracy, Recall, Precision, F1Score

from dataset.utils.data_utils import get_dataset, get_data_loader
from models.audio_models import audio_model_dict
from utils.common_utils import ensure_dir
from utils.train_utils import build_optimizer_and_scheduler
from tqdm.auto import tqdm


class BasePipeline:
    def __init__(self, args, fold: int, accelerator: Accelerator, logger):
        self.args = args
        self.fold = fold
        self.it = 0
        self.accelerator = accelerator
        self.logger = logger
        self.metrics = {'accuracy': Accuracy('binary').cuda(),
                        'precision': Precision('binary').cuda(),
                        'recall': Recall('binary').cuda(),
                        'f1-score': F1Score('binary').cuda()}
        self.device = self.accelerator.device

        self.mode = self.args.mode
        self.model = self.set_model()
        self.save_dir = os.path.join(self.args.model_save_dir, self.args.model_backbone)
        self.dataset_dict = self.set_dataset()
        self.dataloader_dict = self.set_dataloader()
        self.t_total = self.calc_t_total()
        self.optimizer, self.scheduler = self.set_optimizer()
        self.save_metric = self.set_save_metric()
        self.criterion = self.set_criterion()

        self.counter, self.best_metric = torch.tensor(0, dtype=torch.int32).cuda(), 0

    def set_model(self):
        raise NotImplementedError

    def set_dataset(self):
        return get_dataset(self.args, self.fold, self.mode)

    def set_dataloader(self):
        return get_data_loader(self.args, self.dataset_dict)

    def set_criterion(self):
        return nn.CrossEntropyLoss()

    def calc_t_total(self) -> int:
        self.iter_per_epoch = len(self.dataloader_dict['train'])
        return self.args.train_epochs * self.iter_per_epoch

    def set_optimizer(self):
        optimizer, scheduler = build_optimizer_and_scheduler(self.args, self.model, self.t_total)
        return optimizer, scheduler

    def ddp_preparation(self):
        self.model, self.optimizer, self.scheduler, self.dataloader_dict['train'], self.dataloader_dict['val'], \
        self.dataloader_dict['test'] = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.dataloader_dict['train'], self.dataloader_dict['val'],
            self.dataloader_dict['test'])

    def set_save_metric(self) -> str:
        """
        return a metric (e.g. f1-score) that evaluate if the model is the best
        :return:
        """
        return 'f1-score'

    def train(self):
        self.ddp_preparation()

        self.logger.info(
            f'[Fold: {self.fold} / {self.args.k_fold - 1}]: Start Training on {self.accelerator.num_processes} gpus!',
            main_process_only=True)
        self.it = 0

        # early stopping and save configuration
        save_path = os.path.join(self.save_dir, f'fold_{self.fold}.pt')

        for epoch in tqdm(range(self.args.train_epochs), disable=not self.accelerator.is_main_process):
            # training
            train_metrics = self.train_one_epoch(epoch)
            self.logger.info(
                f'[Fold {self.fold} / {self.args.k_fold - 1}, epoch {epoch}], training metrics: {train_metrics}',
                main_process_only=True)
            self.accelerator.log({'train_metrics': train_metrics}, step=self.it)
            self.metrics_reset()

            # evaluation
            val_metrics = self.evaluate(self.dataloader_dict['val'])
            self.accelerator.log({'val_metrics': val_metrics})
            # save best model and early stopping!
            if self.early_stop(val_metrics, save_path, epoch):
                break

        self.logger.info(f'[Fold {self.fold} / {self.args.k_fold - 1}], End Training...', main_process_only=True)
        return self.test()

    def train_one_epoch(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, val_loader):
        raise NotImplementedError

    def early_stop(self, val_metrics, save_path, epoch) -> bool:
        save_metric = val_metrics[self.save_metric]
        if save_metric > self.best_metric:
            self.best_metric = save_metric
            self.accelerator.wait_for_everyone()
            self.save_model(save_path)
            self.counter.zero_()
        else:
            self.counter += 1
        self.logger.info(
            f'[Fold: {self.fold} / {self.args.k_fold - 1}, epoch {epoch}], validation metrics: {val_metrics}, '
            f'counter: {self.counter.item()}', main_process_only=True)
        if self.counter.item() >= self.args.patience:
            self.logger.info(f'[Fold {self.fold} / {self.args.k_fold - 1}, epoch {epoch}], Early Stopping!',
                             main_process_only=True)
            return True
        return False

    def get_save_dict(self):
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        save_dict = {
            'model': unwrapped_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'it': self.it + 1,
        }
        return save_dict

    def save_model(self, save_path):
        ensure_dir(save_path)
        save_dict = self.get_save_dict()

        self.accelerator.save(save_dict, save_path)
        self.logger.info(f"model saved: {save_path}")

    def load_model(self, load_path):
        self.accelerator.wait_for_everyone()
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.logger.info('model loaded')
        return checkpoint

    def metrics_update(self, pred, target):
        for metric in self.metrics.values():
            metric.update(pred, target)

    def metrics_computation(self):
        return_dict = {}
        for key in self.metrics.keys():
            return_dict[key] = round(self.metrics[key].compute().item(), 6)
        return return_dict

    def metrics_reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def update_params(self, loss):
        self.accelerator.backward(loss)
        if self.args.max_grad_norm:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def test(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.save_dir, f'fold_{self.fold}.pt')
        # ensure all processes synchronized
        self.accelerator.wait_for_everyone()
        self.model = self.set_model()
        self.model.load_state_dict(torch.load(save_path)['model'])
        self.model = self.accelerator.prepare_model(self.model)
        # test
        self.metrics_reset()
        test_metrics = self.evaluate(self.dataloader_dict['test'])
        self.logger.info(f'[Fold {self.fold} / {self.args.k_fold - 1}], test metrics: {test_metrics}',
                         main_process_only=True)
        self.accelerator.log({'test_metrics': test_metrics})
        return test_metrics

    @staticmethod
    def get_argument():
        """
        Get specificed arguments into argparse for each pipeline
        """
        return {}
