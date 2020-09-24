import warnings
import numpy as np
warnings.filterwarnings("ignore")


# Loss Meter
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Dice Score for binary segmentation
# https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
class Evaluator(object):
    def __init__(self, target_class=1):
        self.IoU = 0
        self.num_batch = 0
        self.smooth = 1e-6
        self.target_class = target_class

    def iou_fn(self, gt_image, pre_image):
        intersection = (pre_image & gt_image).sum((1, 2))
        union = (pre_image | gt_image).sum((1, 2))

        iou = (intersection + self.smooth) / (union + self.smooth)

        thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
        return iou.mean()


    def add_batch(self, gt_image, pre_image):
        gt_image[gt_image != self.target_class] = 0
        pre_image[pre_image != self.target_class] = 0

        batch_iou = self.iou_fn(gt_image, pre_image)
        self.IoU = (self.IoU * self.num_batch + batch_iou) / (self.num_batch + 1)
        self.num_batch += 1

    def reset(self):
        self.IoU = 0
        self.num_batch = 0
