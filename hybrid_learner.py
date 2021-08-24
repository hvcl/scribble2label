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

from contrastive_loss import contrastive_loss, center_cont_loss, center_cont_loss2, selective_center_cont_loss, selective_cont_loss

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
                        {'params': getattr(model, 'decoder').parameters(), 'lr': self.config.lr * 10},
                        {'params': getattr(model, 'projection_head').parameters(), 'lr':self.config.lr},
                        {'params': getattr(model, 'segmentation_head').parameters(), 'lr': self.config.lr}]

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
            
            # only updates encoder & projection head
            # if self.epoch < self.config.thr_epoch or self.config.mode=='S2L':
            #     for p in self.model.projection_head.parameters():
            #         p.requires_grad = False

            outputs, projections = self.model(images)
            x_loss = self.criterion(outputs, scribbles)

            scribbles = scribbles.cpu()
            mean = weights[..., 0]
            u_labels = torch.where(((mean < (1 - self.config.thr_conf)) |
                                    (mean > self.config.thr_conf)) &
                                    (scribbles == self.config.ignore_index),
                                    mean.round().long(),
                                    self.config.ignore_index * torch.ones_like(scribbles)).to(self.config.device)
            u_loss = self.u_criterion(outputs, u_labels)

            c_loss = contrastive_loss(scribbles, u_labels.cpu(), projections,
                                    self.config.max_nsample, self.config.temperature)
            # c_loss, ctc_loss = center_cont_loss2(scribbles, u_labels.cpu(), projections,
            #                         self.config.max_nsample, self.config.temperature)
            loss = x_loss + u_loss * self.config.alpha_ce + c_loss * self.config.alpha_cont
            # + 0.2 * ctc_loss

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
                
                outputs, projections = self.model(images)
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
                pred = torch.nn.functional.softmax(self.model(img)[0], dim=1)
            weight = torch.tensor(images['weight'])
            pred = pred.squeeze(0).cpu()
            x = pred[1]
            weight[...,0] = self.config.alpha * x + (1-self.config.alpha) * weight[...,0]
            self.train_loader.dataset.images[idx]['weight'] = weight.numpy()
        self.n_ensemble += 1

    def fit(self, epochs):
        for e in range(epochs):
            self.config.alpha_cont = 1 - e/epochs
            self.config.alpha_ce   = e/epochs

            t = time.time()
            loss = self.train_one_epoch()

            self.log(f'[Train] \t Epoch: {self.epoch}, loss: {loss:.5f}, time: {(time.time() - t):.2f}')
            self.tb_log(loss, None, 'Train', self.epoch)

            t = time.time()
            loss, score = self.validation()

            self.log(f'[Valid] \t Epoch: {self.epoch}, loss: {loss:.5f}, IoU: {score:.4f}, time: {(time.time() - t):.2f}')
            self.tb_log(loss, score, 'Valid', self.epoch)
            self.post_processing(loss, score)

            if ((self.epoch + 1) % self.config.period_epoch == 0) and (e > epochs):
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
            self.log(f'best model: {self.best_epoch} epoch - {self.best_score:.4f}')
            self.config.best_score = self.best_score
            self.config.best_epoch = self.best_epoch

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_score = checkpoint['best_score']
        self.epoch = checkpoint['epoch']
        print(f'<model_info>  [best_score] {self.best_score}, [save_epoch] {self.epoch}')
        self.epoch = 0
    
    def log(self, text):
        self.logger.info(text)

    def tb_log(self, loss, IoU, split, step):
        if loss: self.tb_logger.add_scalar(f'{split}/Loss', loss, step)
        if IoU: self.tb_logger.add_scalar(f'{split}/IoU', IoU, step)

    def pseudo_labeling(self):
        ds = self.train_loader.dataset
        transforms = Compose([Normalize(), ToTensorV2()])
        for idx, images in tqdm(ds.images.items(), total=len(ds)):
            augmented = transforms(image=images['image'])
            img = augmented['image'].unsqueeze(0).to(self.config.device)
            with torch.no_grad():
                pred = torch.nn.functional.softmax(self.model(img)[0], dim=1)
            weight = torch.tensor(images['weight'])
            pred = pred.squeeze(0).cpu()
            x = pred[1]
            weight[...,0] = x
            self.train_loader.dataset.images[idx]['weight'] = weight.numpy()
        self.n_ensemble += 1