import os
import sys
from tqdm import tqdm

import cv2
sys.path.append('/root/MVSSM/')
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def center_crop(image, target_width, target_height):
    """
    从图像中心裁剪到指定大小

    参数:
        image: 输入图像 (numpy数组)
        target_width: 目标宽度
        target_height: 目标高度

    返回:
        cropped_image: 裁剪后的图像
    """
    # 获取原始图像的高度和宽度
    height, width = image.shape[:2]

    # 计算裁剪区域的起始坐标
    start_x = max(0, (width - target_width) // 2)
    start_y = max(0, (height - target_height) // 2)

    # 计算裁剪区域的结束坐标
    end_x = min(width, start_x + target_width)
    end_y = min(height, start_y + target_height)

    # 调整起始坐标，确保裁剪区域不会超出图像边界
    start_x = end_x - target_width
    start_y = end_y - target_height

    # 执行裁剪
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image


restor_root = '/root/MVSSM/results/BSD-L'
gt_root = '/root/BSD/test/gt'

psnr = 0.0
ssim = 0.0
count = 0
for video in sorted(os.listdir(restor_root)):
    video_gt_root = gt_root + '/' + video
    video_restor_root = restor_root + '/' + video
    img_gt_list = sorted(os.listdir(video_gt_root))
    img_restor_list = sorted(os.listdir(video_restor_root))
    # if len(img_gt_list) > 100:
    #     img_gt_list = img_gt_list[0:101]
    list_id = 0
    for img in tqdm(img_gt_list, dynamic_ncols=True):
        img_gt_root = video_gt_root + '/' + img_gt_list[list_id]
        img_restor_root = video_restor_root + '/' + img_restor_list[list_id]
        img_gt = cv2.imread(img_gt_root)
        # img_gt = center_crop(img_gt, 1080, 680)
        img_gt = img_gt[:, :, ::-1]
        img_restor = cv2.imread(img_restor_root)[:, :, ::-1]
        # img_restor = center_crop(img_restor, 1080, 680)
        psnr_ = calculate_psnr(img_restor, img_gt, crop_border=0)
        ssim_ = calculate_ssim(img_restor, img_gt, crop_border=0)
        psnr += psnr_
        ssim += ssim_
        count += 1
        list_id += 1

print('PSNR:', psnr / count)
print('SSIM:', ssim / count)
