import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from scripts.dataset import get_transforms, dsbTestDataset
from segmentation_models_pytorch.unet import Unet
from scripts.metric_mdice import Evaluator as mdice_evaluator
from scripts.metric import Evaluator as iou_evaluator


class config:
    seed = 42
    name = 'MoNuSeg' # bright_field, histopathology, fluorescence, MoNuSeg, EM
    device = torch.device('cuda:0')
    proj_ch = 32 # 32, 16
    fold = 3 # 0~3
    scr = 'manual'
    temp = 1.0
    save_result = True
    """ Path """
    data_dir = f'/workspace/Dataset/examples/images/{name}/'
    mask_dir = f'/workspace/Dataset/examples/labels/{name}/full/'
    df_path = f'/workspace/Dataset/examples/labels/{name}/test.csv'
    model_path = f'/workspace/scribble2label/logs/{name}/proj_ch{proj_ch}/fold{fold}_scr_{scr}/best_model.pth'#temp{int(temp*100):03d}/best_model.pth'
    """ Testing """
    input_size = 256
    batch_size = 1
    num_workers = 8


def inference_image(net, images):
    with torch.no_grad():
        predictions, projections = net(images)
        predictions = F.softmax(predictions, dim=1)
    return predictions.detach().cpu().numpy()


def inference(net, test_loader, save_dir=None):
    semantic_eval, instance_eval = iou_evaluator(), mdice_evaluator()
    semantic_eval.reset()
    instance_eval.reset()
    for image_names, images, masks in tqdm(test_loader):
        images = images.to(config.device)
        masks = masks.numpy()
        predictions = inference_image(net, images)
        predictions = np.argmax(predictions, axis=1).astype('uint8')
        semantic_eval.add_batch((masks > 0).astype('uint8'), predictions)
        for image_name, pred, mask in zip(image_names, predictions, masks):
            instance_eval.add_pred(mask, pred)
            if save_dir:
                Image.fromarray(pred * 255).save(os.path.join(save_dir, f'{image_name}.png'))
    return semantic_eval.IoU, instance_eval.Dice


if __name__ == '__main__':
    # model = Unet(encoder_name='resnet50', encoder_weights='imagenet', decoder_use_batchnorm=True,
    #              decoder_attention_type='scse', classes=2, activation=None)
    if config.name=='histopathology' or 'MoNuSeg':
        model = model.UnetCustom(encoder_name='resnet34', encoder_weights='imagenet', projection_ch=config.proj_ch,
                                decoder_attention_type='scse', activation=None, classes=2)
    else:
        model = model.UnetCustom(encoder_name='resnet50', encoder_weights='imagenet', projection_ch=config.proj_ch,
                            decoder_attention_type='scse', activation=None, classes=2)

    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    print(f'Model Loaded: {config.model_path}')

    test_df = pd.read_csv(config.df_path)
    transforms = get_transforms(config.input_size, need=('val'))

    test_dataset = dsbTestDataset(config.data_dir, config.mask_dir, test_df,
                                  tfms=transforms['val'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                             shuffle=False, sampler=None, pin_memory=True)

    if config.save_result:
        save_dir = os.path.join(os.path.dirname(config.model_path), 'predictions')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    iou, mdice = inference(model, test_loader, save_dir)
    print(f'IoU: {iou}, mDice: {mdice}')
