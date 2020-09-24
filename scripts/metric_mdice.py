import warnings
import numpy as np
import cv2
import skimage.measure as measure
import skimage.morphology as morphology
from scipy.ndimage.morphology import binary_fill_holes
warnings.filterwarnings("ignore")

# Dice Score & IoU for binary segmentation
class Evaluator(object):
    def __init__(self):
        self.Dice = 0
        self.IoU = 0
        self.num_batch = 0
        self.eps = 1e-4

    def dice_fn(self, gt_image, pre_image):
        eps = 1e-4
        batch_size = pre_image.shape[0]

        pre_image = pre_image.reshape(batch_size, -1).astype(np.bool)
        gt_image = gt_image.reshape(batch_size, -1).astype(np.bool)

        intersection = np.logical_and(pre_image, gt_image).sum(axis=1)
        union = pre_image.sum(axis=1) + gt_image.sum(axis=1) + eps
        Dice = ((2. * intersection + eps) / union).mean()
        IoU = Dice / (2. - Dice)
        return Dice, IoU

    # https://github.com/naivete5656/WSISPDR/blob/master/utils/for_review.py
    def mdice_fn(self, target, pred):
        '''
        :param target: hxw label
        :param pred: hxw label
        :return: mIoU, mDice
        '''
        iou_mean = 0.
        dice_mean = 0.
        for idx, target_label in enumerate(range(1, target.max() + 1)):
            if np.sum(target == target_label) < 20:
                target[target == target_label] = 0
                # seek pred label correspond to the label of target
            correspond_labels = pred[target == target_label]
            correspond_labels = correspond_labels[correspond_labels != 0]
            unique, counts = np.unique(correspond_labels, return_counts=True)
            try:
                max_label = unique[counts.argmax()]
                pred_mask = np.zeros(pred.shape)
                pred_mask[pred == max_label] = 1
            except ValueError:
                bou_list = []
                max_bou = target.shape[0]
                max_bou_h = target.shape[1]
                bou_list.extend(target[0, :])
                bou_list.extend(target[max_bou - 1, :])
                bou_list.extend(target[:, max_bou_h - 1])
                bou_list.extend(target[:, 0])
                bou_list = np.unique(bou_list)
                for x in bou_list:
                    target[target == x] = 0
                pred_mask = np.zeros(pred.shape)

            # create mask
            target_mask = np.zeros(target.shape)
            target_mask[target == target_label] = 1
            pred_mask = pred_mask.flatten()
            target_mask = target_mask.flatten()

            tp = pred_mask.dot(target_mask)
            fn = pred_mask.sum() - tp
            fp = target_mask.sum() - tp

            iou = ((tp + self.eps) / (tp + fp + fn + self.eps))
            dice = (2 * tp + self.eps) / (2 * tp + fn + fp + self.eps)
            iou_mean = (iou_mean * idx + iou) / (idx + 1)
            dice_mean = (dice_mean * idx + dice) / (idx + 1)
        return dice_mean, iou_mean

    def add_pred(self, gt_image, pre_image):
        pre_image = measure.label(pre_image)
        pre_image = morphology.remove_small_objects(pre_image, min_size=64)
        pre_image = binary_fill_holes(pre_image > 0)
        pre_image = measure.label(pre_image)

        batch_mdice, batch_miou = self.mdice_fn(gt_image, pre_image)
        self.Dice = (self.Dice * self.num_batch + batch_mdice) / (self.num_batch + 1)
        self.IoU = (self.IoU * self.num_batch + batch_miou) / (self.num_batch + 1)
        self.num_batch += 1

    def reset(self):
        self.Dice = 0
        self.IoU = 0
        self.num_batch = 0
