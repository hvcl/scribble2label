# Clustering Code : https://www.kaggle.com/mpware/stage1-eda-microscope-image-types-clustering

import numpy as np
import pandas as pd
import skimage.io
import os
import shutil
import random

from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from skimage.measure import label
from skimage.morphology import skeletonize
from skimage.feature import corner_harris, corner_peaks

import warnings
warnings.filterwarnings('ignore')

STAGE1_TRAIN = "./examples/raw_data"
STAGE1_TRAIN_IMAGE_PATTERN = "%s/{}/images/{}.png" % STAGE1_TRAIN
STAGE1_TRAIN_MASK_PATTERN = "%s/{}/masks/*.png" % STAGE1_TRAIN
IMAGE_ID = "image_id"
IMAGE_WIDTH = "width"
IMAGE_WEIGHT = "height"
HSV_CLUSTER = "hsv_cluster"
HSV_DOMINANT = "hsv_dominant"
TOTAL_MASK = "total_masks"


def image_ids_in(root_dir, ignore=[]):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


def read_image(image_id, space="rgb"):
    image_file = STAGE1_TRAIN_IMAGE_PATTERN.format(image_id, image_id)
    image = skimage.io.imread(image_file)
    # Drop alpha which is not used
    image = image[:, :, :3]
    if space == "hsv":
        image = skimage.color.rgb2hsv(image)
    return image


# Get image width, height and count masks available.
def read_image_labels(image_id, space="rgb"):
    image = read_image(image_id, space = space)
    mask_file = STAGE1_TRAIN_MASK_PATTERN.format(image_id)
    masks = skimage.io.imread_collection(mask_file).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1 #255
    return image, labels, num_masks


def get_domimant_colors(img, top_colors=2):
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters = top_colors)
    clt.fit(img_l)
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return clt.cluster_centers_, hist


def get_images_details(image_ids):
    details = []
    for image_id in image_ids:
        image_hsv, labels, num_masks = read_image_labels(image_id, space="hsv")
        height, width, l = image_hsv.shape
        dominant_colors_hsv, dominant_rates_hsv = get_domimant_colors(image_hsv, top_colors=1)
        dominant_colors_hsv = dominant_colors_hsv.reshape(1, dominant_colors_hsv.shape[0] * dominant_colors_hsv.shape[1])
        info = (image_id, width, height, num_masks, dominant_colors_hsv.squeeze())
        details.append(info)
    return details


def remove_corner(mask, coords):
    for coord in coords:
        x, y = coord
        mask[x-1][y-1] = 0
        mask[x-1][y] = 0
        mask[x-1][y+1] = 0
        mask[x][y-1] = 0
        mask[x][y] = 0
        mask[x][y+1] = 0
        mask[x+1][y-1] = 0
        mask[x+1][y] = 0
        mask[x+1][y+1] = 0
    return mask


def scribblize(mask, ratio=0.1):
    """
    Automatically generate scribble-label with a specific ratio.
    the number of foreground scribbles is same with the number of background scribbles
    :param mask: fully annotated label
    :param ratio: scribble ratio
    :return: foreground & background scribbles
    """
    sk = skeletonize(mask)

    i_mask = np.abs(mask - 1) // 255
    i_sk = skeletonize(i_mask)
    coords = corner_peaks(corner_harris(i_sk), min_distance=5)
    i_sk = remove_corner(i_sk, coords)

    label_sk = label(sk)
    n_sk = np.max(label_sk)
    n_remove = int(n_sk * (1-ratio))
    removes = random.sample(range(1, n_sk+1), n_remove)
    for i in removes:
        label_sk[label_sk == i] = 0
    sk = (label_sk > 0).astype('uint8')

    label_i_sk = label(i_sk)
    n_i_sk = np.max(label_i_sk)
    n_i_remove = n_i_sk - (n_sk - n_remove)
    removes = random.sample(range(1, n_i_sk+1), n_i_remove)
    for i in removes:
        label_i_sk[label_i_sk == i] = 0
    i_sk = (label_i_sk > 0).astype('uint8')
    return sk, i_sk


if __name__ == '__main__':
    # Load stage 1 image identifiers.
    train_image_ids = image_ids_in(STAGE1_TRAIN)
    META_COLS = [IMAGE_ID, IMAGE_WIDTH, IMAGE_WEIGHT, TOTAL_MASK]
    COLS = META_COLS + [HSV_DOMINANT]

    details = get_images_details(train_image_ids)

    trainPD = pd.DataFrame(details, columns=COLS)
    X = (pd.DataFrame(trainPD[HSV_DOMINANT].values.tolist()))
    kmeans = KMeans(n_clusters=3).fit(X)
    clusters = kmeans.predict(X)
    trainPD[HSV_CLUSTER] = clusters

    ratios = [0.1, 0.3, 0.5, 1.0]

    image_types = ['fluorescence', 'histopathology', 'bright_field']
    for idx_type, image_type in enumerate(image_types):
        os.makedirs(f'./examples/images/{image_type}', exist_ok=True)
        os.makedirs(f'./examples/labels/{image_type}', exist_ok=True)
        for image_name in trainPD[trainPD[HSV_CLUSTER] == idx_type][IMAGE_ID].values:
            os.makedirs(f'./examples/labels/{image_type}/full', exist_ok=True)
            shutil.copyfile(f'./examples/raw_data/{image_name}/images/{image_name}.png',
                            f'./examples/images/{image_type}/{image_name}.png')
            image, labels, num_masks = read_image_labels(image_name)
            skimage.io.imsave(f'./examples/labels/{image_type}/full/{image_name}.png', labels)
            for ratio in ratios:
                os.makedirs(f'./examples/labels/{image_type}/scribble{int(ratio * 100)}', exist_ok=True)
                mask = (labels > 0).astype('uint8')
                sk, i_sk = scribblize(mask, ratio=ratio)
                scr = np.ones_like(mask) * 250
                scr[i_sk == 1] = 0
                scr[sk == 1] = 1
                skimage.io.imsave(f'./examples/labels/{image_type}/scribble{int(ratio * 100)}/{image_name}.png', scr)

    for idx_type, image_type in enumerate(image_types):
        dataset_info = {
            'ImageID': trainPD[trainPD[HSV_CLUSTER] == idx_type][IMAGE_ID].values
        }
        dataset_info = pd.DataFrame(dataset_info)
        kf = KFold(n_splits=5, shuffle=True)

        dataset_info.loc[:, 'fold'] = 0
        for fold_number, (train_index, val_index) in enumerate(kf.split(X=dataset_info.index)):
            dataset_info.loc[dataset_info.iloc[val_index].index, 'fold'] = fold_number
        train_info = dataset_info[dataset_info.fold != 4].reset_index(drop=True)
        test_info = dataset_info[dataset_info.fold == 4].reset_index(drop=True)

        train_info.to_csv(f'./examples/labels/{image_type}/train.csv', index=False)
        test_info.to_csv(f'./examples/labels/{image_type}/test.csv', index=False)
