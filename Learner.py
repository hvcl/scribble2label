import os
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from scripts.utils import init_logger
from scripts.tb_utils import init_tb_logger
from scripts.metric import Evaluator, AverageMeter
from scripts.optimizer import RAdam

from albumentations.core.composition import Compose
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensorV2


class Learner:
    def __init__(self, model, train_loader, valid_loader, config):
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model.to(self.config.device)

        self.logger = init_logger(self.config.log_dir, 'train_main.log')
        self.tb_logger = init_tb_logger(self.config.log_dir, 'train_main')
        self.log('\n'.join([f"{k} = {v}" for k, v in self.config.__dict__.items()]))

        self.summary_loss = AverageMeter()
        self.evaluator = Evaluator()

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        self.u_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        train_params = [{'params': getattr(model, 'encoder').parameters(), 'lr': self.config.lr},
                        {'params': getattr(model, 'decoder').parameters(), 'lr': self.config.lr * 10}]
        self.optimizer = RAdam(train_params, weight_decay=self.config.weight_decay)

        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=2, T_mult=2, eta_min=1e-6)

        self.n_ensemble = 0
        self.epoch = 0
        self.best_epoch = 0
        self.best_loss = np.inf
        self.best_score = -np.inf

    def train_one_epoch(self):
        self.model.train()
        self.summary_loss.reset()
        iters = len(self.train_loader)
        for step, (images, scribbles, weights) in enumerate(self.train_loader):
            self.tb_logger.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'],
                                      iters * self.epoch + step)
            scribbles = scribbles.to(self.config.device).long()
            images = images.to(self.config.device)
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            if self.epoch < self.config.thr_epoch:
                loss = self.criterion(outputs, scribbles)
            else:
                x_loss = self.criterion(outputs, scribbles)

                scribbles = scribbles.cpu()
                mean = weights[..., 0]
                u_labels = torch.where(((mean < (1 - self.config.thr_conf)) |
                                        (mean > self.config.thr_conf)) &
                                       (scribbles == self.config.ignore_index),
                                       mean.round().long(),
                                       self.config.ignore_index * torch.ones_like(scribbles)).to(self.config.device)
                u_loss = self.u_criterion(outputs, u_labels)
                loss = x_loss + 0.5 * u_loss

            loss.backward()
            self.summary_loss.update(loss.detach().item(), batch_size)
            self.optimizer.step()
            if self.scheduler.__class__.__name__ != 'ReduceLROnPlateau':
                self.scheduler.step()

        return self.summary_loss.avg

    def validation(self):
        self.model.eval()
        self.summary_loss.reset()
        self.evaluator.reset()
        for step, (_, images, _, targets) in enumerate(self.valid_loader):
            with torch.no_grad():
                targets = targets.to(self.config.device).long()
                batch_size = images.shape[0]
                images = images.to(self.config.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                targets = targets.cpu().numpy()
                outputs = torch.argmax(outputs, dim=1)
                outputs = outputs.data.cpu().numpy()
                self.evaluator.add_batch(targets, outputs)
                self.summary_loss.update(loss.detach().item(), batch_size)

        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(self.evaluator.IoU)
        return self.summary_loss.avg, self.evaluator.IoU

    def ensemble_prediction(self):
        ds = self.train_loader.dataset
        transforms = Compose([Normalize(), ToTensorV2()])
        for idx, images in tqdm(ds.images.items(), total=len(ds)):
            augmented = transforms(image=images['image'])
            img = augmented['image'].unsqueeze(0).to(self.config.device)
            with torch.no_grad():
                pred = torch.nn.functional.softmax(self.model(img), dim=1)
            weight = torch.tensor(images['weight'])
            pred = pred.squeeze(0).cpu()
            x = pred[1]
            weight[...,0] = self.config.alpha * x + (1-self.config.alpha) * weight[...,0]
            self.train_loader.dataset.images[idx]['weight'] = weight.numpy()
        self.n_ensemble += 1

    def fit(self, epochs):
        for e in range(epochs):
            t = time.time()
            loss = self.train_one_epoch()

            self.log(f'[Train] \t Epoch: {self.epoch}, loss: {loss:.5f}, time: {(time.time() - t):.2f}')
            self.tb_log(loss, None, 'Train', self.epoch)

            t = time.time()
            loss, score = self.validation()

            self.log(f'[Valid] \t Epoch: {self.epoch}, loss: {loss:.5f}, IoU: {score:.4f}, time: {(time.time() - t):.2f}')
            self.tb_log(loss, score, 'Valid', self.epoch)
            self.post_processing(loss, score)

            if (self.epoch + 1) % self.config.period_epoch == 0:
                self.log(f'[Ensemble] \t the {self.n_ensemble}th Prediction Ensemble ...')
                self.ensemble_prediction()

            self.epoch += 1
        self.log(f'best epoch: {self.best_epoch}, best loss: {self.best_loss}, best_score: {self.best_score}')

    def post_processing(self, loss, score):
        if loss < self.best_loss:
            self.best_loss = loss

        if score > self.best_score:
            self.best_score = score
            self.best_epoch = self.epoch

            self.model.eval()
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_score': self.best_score,
                'epoch': self.epoch,
            }, f'{os.path.join(self.config.log_dir, "best_model.pth")}')
            self.log(f'best model: {self.epoch} epoch - {score:.4f}')

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_score = checkpoint['best_score']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, text):
        self.logger.info(text)

    def tb_log(self, loss, IoU, split, step):
        if loss: self.tb_logger.add_scalar(f'{split}/Loss', loss, step)
        if IoU: self.tb_logger.add_scalar(f'{split}/IoU', IoU, step)
