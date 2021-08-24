# public
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import requests

# custom
import model
from scripts.dataset import get_transforms, dsbDataset
from segmentation_models_pytorch.unet import Unet
from scripts.utils import seed_everything
from hybrid_learner import Learner

# Line notify
TARGET_URL = 'https://notify-api.line.me/api/notify'
TOKEN = 'XluqWRqZPUq2UOh8bRxnlPjl9gLyQp9YhgVuxKRGGeM'

class config:
    seed = 42
    name = f'MoNuSeg'  # bright_field, histopathology, fluorescence, MoNuSeg, EM
    device = torch.device('cuda:0')
    mode = 'cont' # S2L, cont
    scr = 'manual' # 10 30 50 100 manual
    exp_setting = 'default' # center01 select 
    fold = 3
    proj_ch = 32 # 16, 32, 64
    Lam = 1.0 # 0.5: default
    if exp_setting=='default':
        Lam = 0.5
        proj_ch = 32
    """ Path """
    data_dir = f'/workspace/Dataset/examples/images/{name}/'
    mask_dir = f'/workspace/Dataset/examples/labels/{name}/full/'
    df_path = f'/workspace/Dataset/examples/labels/{name}/train.csv'
    scr_dir = f'/workspace/Dataset/examples/labels/{name}/scribble_{scr}/'
    stage1_log = f'/workspace/scribble2label/logs/{name}/proj_ch{proj_ch}/fold{fold}_scr_{scr}'
    log_dir = f'/workspace/scribble2label/logs/{name}/proj_ch{proj_ch}/fold{fold}_scr_{scr}/stage2'
    """ Training """
    alpha_cont = 1
    alpha_ce = 0    
    n_epochs = 2000
    input_size = 256
    batch_size = 30
    lr = 3e-4
    weight_decay = 5e-5
    num_workers = 8
    ignore_index = 250
    """ Scribble2Label Params """
    # thr_epoch = 100
    period_epoch = 5
    if name=='MoNuSeg':
        thr_conf = 0.95
        alpha = 0.1
    else:
        thr_conf = 0.8
        alpha = 0.2
    """ Contrastive Params """
    if mode=='cont':
        temperature = 0.1
        max_nsample = 6000
        log_dir += f'_temp{int(temperature*100):03d}_nsample{max_nsample}'
    """ For Line Message """
    best_score = 0
    best_epoch = 0
    exp_name = log_dir.split('logs/')[-1]


if __name__ == '__main__':
    seed_everything(config.seed)
    if config.name == 'histopahology' or 'MoNuSeg': 
        # resnet34: histopahology MoNuSeg
        if config.exp_setting=='default':
            model = model.UnetCustom(encoder_name='resnet34' ,encoder_weights='imagenet', projection_ch=config.proj_ch,
                                    decoder_attention_type='scse', activation=None, classes=2)
        else:
            model = model.UnetModified(encoder_name='resnet34' ,encoder_weights='imagenet', projection_ch=config.proj_ch,
                                    decoder_attention_type='scse', activation=None, classes=2)
    else: 
        # resnet50: bright_field fluorescence
        if config.exp_setting=='default':
            model = model.UnetCustom(encoder_name='resnet50', encoder_weights='imagenet', projection_ch=config.proj_ch,
                                    decoder_attention_type='scse', activation=None, classes=2)
        else:
            model = model.UnetModified(encoder_name='resnet50', encoder_weights='imagenet', projection_ch=config.proj_ch,
                                    decoder_attention_type='scse', activation=None, classes=2)

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
    # Load stage1 and do labeling
    pretrained_path = os.path.join(config.stage1_log, 'best_model.pth')
    if os.path.isfile(pretrained_path):
        Learner.load(pretrained_path)
        Learner.log(f"Checkpoint Loaded: {pretrained_path}")        
    Learner.pseudo_labeling()
    Learner.fit(config.n_epochs)

    response = requests.post(
        TARGET_URL,
        headers={
            'Authorization': 'Bearer ' + TOKEN
        },
        data={
            'message': f'\nExperiment result \
                         \nSetting: {config.exp_name} \
                         \nScore: {config.best_score} \
                         \nEpoch: {config.best_epoch}'
        }
    )