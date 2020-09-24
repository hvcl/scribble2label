import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.dataset import get_transforms, dsbDataset
from segmentation_models_pytorch.unet import Unet
from scripts.utils import seed_everything
from Learner import Learner


class config:
    seed = 42
    name = f'fluorescence'
    device = torch.device('cuda:0')
    """ Path """
    data_dir = f'./examples/images/{name}/'
    scr_dir = f'./examples/labels/{name}/scribble30/'
    mask_dir = f'./examples/labels/{name}/full/'
    df_path = f'./examples/labels/{name}/train.csv'
    log_dir = f'./logs/{name}'
    """ Training """
    fold = 0
    n_epochs = 10000
    input_size = 256
    batch_size = 30
    lr = 3e-4
    weight_decay = 5e-5
    num_workers = 8
    ignore_index = 250
    """ Scribble2Label Params """
    thr_epoch = 100
    period_epoch = 5
    thr_conf = 0.8
    alpha = 0.2


if __name__ == '__main__':
    seed_everything(config.seed)

    model = Unet(encoder_name='resnet50', encoder_weights='imagenet', decoder_use_batchnorm=True,
                 decoder_attention_type='scse', classes=2, activation=None)

    df = pd.read_csv(config.df_path)
    train_df = df[df.fold != config.fold].reset_index(drop=True)
    valid_df = df[df.fold == config.fold].reset_index(drop=True)
    transforms = get_transforms(config.input_size, need=('train', 'val'))

    train_dataset = dsbDataset(config.data_dir, config.scr_dir, config.mask_dir, train_df,
                               tfms=transforms['train'], return_id=False)
    valid_dataset = dsbDataset(config.data_dir, config.scr_dir, config.mask_dir, valid_df,
                               tfms=transforms['val'], return_id=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, num_workers=config.num_workers,
                              shuffle=False)

    Learner = Learner(model, train_loader, valid_loader, config)
    pretrained_path = os.path.join(config.log_dir, 'best_model.pth')
    if os.path.isfile(pretrained_path):
        Learner.load(pretrained_path)
        Learner.log(f"Checkpoint Loaded: {pretrained_path}")
    Learner.fit(config.n_epochs)
