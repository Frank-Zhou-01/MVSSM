import os

import cv2
import numpy as np
from tqdm import tqdm

root = '/root/MVSSM/results'

dirs = os.listdir(root + '/MVSSML-BSD-Test/visualization/BSD-O/')
for d in dirs:
    images = os.listdir(root + '/MVSSML-BSD-Test/visualization/BSD-O/' + d)
    for img in tqdm(images, dynamic_ncols=True):
        img1 = cv2.imread(root + '/MVSSML-BSD-Test/visualization/BSD-O/' + d + '/' + img)
        img2 = cv2.imread(root + '/MVSSML-BSD-Test/visualization/BSD-90/' + d + '/' + img)
        img3 = cv2.imread(root + '/MVSSML-BSD-Test/visualization/BSD-180/' + d + '/' + img)
        img4 = cv2.imread(root + '/MVSSML-BSD-Test/visualization/BSD-270/' + d + '/' + img)
        img5 = cv2.imread(root + '/MVSSML-BSD-Test/visualization/BSD-V/' + d + '/' + img)
        img6 = cv2.imread(root + '/MVSSML-BSD-Test/visualization/BSD-H/' + d + '/' + img)
        img7 = cv2.imread(root + '/MVSSML-BSD-Test/visualization/BSD-T/' + d + '/' + img)
        print(img1.shape, img2.shape, img3.shape, img4.shape, img5.shape, img6.shape, img7.shape)

        # img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # img3 = cv2.rotate(img3, cv2.ROTATE_180)
        # img4 = cv2.rotate(img4, cv2.ROTATE_90_CLOCKWISE)
        # img5 = cv2.flip(img5, 0)
        # img6 = cv2.flip(img6, 1)
        # img7 = cv2.transpose(img7)

        imgs = np.stack([img1, img2, img3, img4, img5, img6, img7], axis=0)
        imgs = np.mean(imgs, axis=0)
        imgs = imgs.astype(np.uint8)

        save_path = root + '/BSD-L-mean/' + d
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path + '/' + img, imgs)
