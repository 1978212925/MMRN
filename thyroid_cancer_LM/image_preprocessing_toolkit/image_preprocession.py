import os
import random

import cv2
import numpy as np


def img_mask_name_standaradization(img_root_path, start_index, end_index):
    mask_root_path = img_root_path.replace('img', 'mask')
    img_name_list = os.listdir(img_root_path)

    for name in img_name_list:
        name_stand = name[start_index: end_index]
        img_old_content_path = os.path.join(img_root_path, name)
        img_stand_content_path = os.path.join(img_root_path, name_stand + '.png')
        mask_old_content_path = os.path.join(mask_root_path, name)
        mask_stand_content_path = os.path.join(mask_root_path, name_stand + '.png')

        os.rename(img_old_content_path, img_stand_content_path)
        os.rename(mask_old_content_path, mask_stand_content_path)
        print(f'rename {name} to {name_stand}.png')


def check_img_size(img_root_path, img_size):
    mask_root_path = img_root_path.replace('img', 'mask')
    img_name_list = os.listdir(img_root_path)

    for name in img_name_list:
        img_content_path = os.path.join(img_root_path, name)
        mask_content_path = os.path.join(mask_root_path, name)
        img = cv2.imread(img_content_path)
        mask = cv2.imread(mask_content_path)
        if img.shape[0] != img_size or img.shape[1] != img_size:
            print(f'img pixel is not standard path: {img_content_path}')
        if mask.shape[0] != img_size or mask.shape[1] != img_size:
            print(f'mask pixel is not standard path: {mask_content_path}')


def img_size_standaradization(img_root_path, img_size):
    mask_root_path = img_root_path.replace('img', 'mask')
    img_name_list = os.listdir(img_root_path)
    for name in img_name_list:
        img_content_path = os.path.join(img_root_path, name)
        mask_content_path = os.path.join(mask_root_path, name)
        img = cv2.imread(img_content_path)
        mask = cv2.imread(mask_content_path)
        # 获取图像大小
        height, width, _ = img.shape
        # 如果图像尺寸大于512*512，缩放为512*512
        if height > img_size or width > img_size:
            resize_img = cv2.resize(img, (img_size,img_size))
            resize_mask = cv2.resize(mask, (img_size,img_size))
            # 将修改的图片保存回去
            cv2.imwrite(img_content_path, resize_img)
            cv2.imwrite(mask_content_path, resize_mask)
        # 如果图像尺寸小于512*512，填充为512*512
        elif height < img_size or width < img_size:
            # 创建一个512*512的黑色背景图像
            background_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            background_mask = np.zeros((img_size, img_size, 3), dtype=np.uint8)

            # 计算填充的起始位置
            start_h = (img_size - height) // 2
            start_w = (img_size - width) // 2

            # 将图像放置在背景图像中心
            background_img[start_h:start_h + height, start_w:start_w + width] = img
            background_mask[start_h:start_h + height, start_w:start_w + width] = mask
            cv2.imwrite(img_content_path, background_img)
            cv2.imwrite(mask_content_path, background_mask)
        elif height == img_size and width == img_size:
            pass

def mask_pixel_standaradization(mask_root_path):
    mask_name_list = os.listdir(mask_root_path)
    for name in mask_name_list:
        mask_content_path = os.path.join(mask_root_path, name)
        # 读取输入图片
        mask = cv2.imread(mask_content_path)
        # 将大于200的像素改成255，将小于200的改成0
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        # cv2.imshow('1', mask)
        # cv2.waitKey(-1)
        # 把修改后的图像数据写回原文件
        cv2.imwrite(mask_content_path, mask)
    print('all mask pixel are standardized')

def check_img_mask_size_is_equal(img_root_path):
    mask_root_path = img_root_path.replace('img', 'mask')
    img_name_list = os.listdir(img_root_path)
    for name in img_name_list:
        img_content_path = os.path.join(img_root_path, name)
        mask_content_path = os.path.join(mask_root_path, name)
        img = cv2.imread(img_content_path)
        mask = cv2.imread(mask_content_path)
        # 获取图像大小
        img_height, img_width, _ = img.shape
        mask_height, mask_width, _ = mask.shape

        if img_height != mask_height or img_width != mask_width:
            print(img_content_path)
        else:
            pass

def crop_img_mask(img_root_path, img_size, crop_size):
    crop_size = crop_size // 2
    mask_root_path = img_root_path.replace('img', 'mask')
    img_name_list = os.listdir(img_root_path)
    for name in img_name_list:
        flag = np.zeros((crop_size, crop_size), dtype=int)
        img_content_path = os.path.join(img_root_path, name)
        mask_content_path = os.path.join(mask_root_path, name)
        img = cv2.imread(img_content_path, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_content_path, flags=cv2.IMREAD_GRAYSCALE)
        x = []
        y = []
        for i in range(img_size):
            for j in range(img_size):
                if (mask[i, j] != 0) & (flag[i, j] == 0):
                    y.append(i)
                    x_count = 0
                    x_mid = 0
                    a = 0
                    while mask[i, j + a] == 1:
                        flag[i][j + a] = 0
                        x_count = x_count + 1
                        a += 1
                    x_mid = x_count / 2
                    x.append(j + x_mid)

        y_final = int(np.mean(y))
        x_final = int(np.mean(x))
        while y_final - crop_size < 0 or y_final + crop_size >= 512 or x_final - crop_size < 0 or x_final + crop_size >= 512:
            if y_final - crop_size < 0:
                y_final += 1
            elif y_final + crop_size >= img_size:
                y_final -= 1
            elif x_final - crop_size < 0:
                x_final += 1
            elif x_final + crop_size >= img_size:
                x_final -= 1
        print(f'img {name} cropped center is [{x_final}, {y_final}]')
        crop_img = img[y_final - crop_size:y_final + crop_size, x_final - crop_size:x_final + crop_size]
        crop_mask = mask[y_final - crop_size:y_final + crop_size, x_final - crop_size:x_final + crop_size]
        cv2.imwrite(img_content_path.replace("img", "cropped_img"), crop_img)
        cv2.imwrite(mask_content_path.replace("mask", "cropped_mask"), crop_mask)


def random_split_cohort(img_root_path, output_img_root_path, split_rario=0.2, random_seed=120):
    img_name_list = os.listdir(img_root_path)
    mask_root_path = img_root_path.replace('img', 'mask')
    output_mask_root_path = output_img_root_path.replace('img', 'mask')
    random.seed(random_seed)
    random.shuffle(img_name_list)
    for i in range(len(img_name_list)):
        img_content_path = os.path.join(img_root_path, img_name_list[i])
        mask_content_path = os.path.join(mask_root_path, img_name_list[i])
        output_img_content_path = os.path.join(output_img_root_path, img_name_list[i])
        output_mask_content_path = os.path.join(output_mask_root_path, img_name_list[i])
        if i < int(len(img_name_list) * split_rario):
            os.replace(img_content_path, output_img_content_path)
            os.replace(mask_content_path, output_mask_content_path)
        else:
            pass
    print(f'split complete-----------')
    print(f'{img_root_path} len:{len(os.listdir(img_root_path)), len(os.listdir(mask_root_path))}')
    print(f'{output_img_root_path} len:{len(os.listdir(output_img_root_path)), len(os.listdir(output_mask_root_path))}')


def get_PIM(cohort_name_list, cohort_root_path):
    for cohort_name in cohort_name_list:
        cohort_content_path = os.path.join(cohort_root_path, cohort_name)
        img_root_path = os.path.join(cohort_content_path, 'cropped_img')
        mask_root_path = os.path.join(cohort_content_path, 'cropped_mask')
        img_name_list = os.listdir(img_root_path)
        for img_name in img_name_list:
            img_content_path = os.path.join(img_root_path, img_name)
            mask_content_path = os.path.join(mask_root_path, img_name)
            img = cv2.imread(img_content_path, flags=cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_content_path, flags=cv2.IMREAD_GRAYSCALE)
            mask = np.where(mask > 200, 255, 0).astype(np.uint8)
            kernel = np.ones(shape=[3, 3], dtype=np.uint8)
            expansion = cv2.dilate(mask, kernel, iterations=4)
            mask = (mask / 255).astype(np.uint8)
            expansion = (expansion / 255).astype(np.uint8)

            peritumor = img * (expansion - mask)
            intratumor = img * mask
            mergeRegion = peritumor + intratumor

            peritumor_path = os.path.join(f"../dataset/peritumor/{cohort_name}", img_name)
            intratumor_path = os.path.join(f"../dataset/intratumor/{cohort_name}", img_name)
            mergeRegion_path = os.path.join(f"../dataset/merge_region/{cohort_name}", img_name)

            cv2.imwrite(peritumor_path, peritumor)
            cv2.imwrite(intratumor_path, intratumor)
            cv2.imwrite(mergeRegion_path, mergeRegion)

def get_fat_roi(cohort_name_list, cohort_root_path):
    for cohort_name in cohort_name_list:
        cohort_content_path = os.path.join(cohort_root_path, cohort_name)
        img_root_path = os.path.join(cohort_content_path, 'cropped_img')
        mask_root_path = os.path.join(cohort_content_path, 'cropped_mask')
        img_name_list = os.listdir(img_root_path)
        for img_name in img_name_list:
            img_content_path = os.path.join(img_root_path, img_name)
            mask_content_path = os.path.join(mask_root_path, img_name)
            img = cv2.imread(img_content_path, flags=cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_content_path, flags=cv2.IMREAD_GRAYSCALE)
            mask = np.where(mask > 200, 255, 0).astype(np.uint8)
            mask = (mask / 255).astype(np.uint8)

            fat_roi = img * mask

            intratumor_path = os.path.join(f"../dataset/{cohort_name}", img_name)

            cv2.imwrite(intratumor_path, fat_roi)
if __name__ == '__main__':
    output_img_root_path = './internal_test_cohort/cropped_img'
    img_root_path = './primary_cohort/cropped_img'
    mask_root_path = './primary_cohort/cropped_mask'
    cohort_name_list = ['primary_cohort', 'internal_test_cohort']
    cohort_root_path = '../image_preprocessing_toolkit'
    # check_img_size(img_root_path, 512)
    # mask_pixel_standaradization(mask_root_path)
    # random_split_cohort(img_root_path, output_img_root_path, split_rario=0.23)
    # get_PIM(cohort_name_list=cohort_name_list, cohort_root_path=cohort_root_path)
    # check_img_mask_size_is_equal(img_root_path)
    # img_size_standaradization(img_root_path, 512)
    # get_fat_roi(cohort_name_list, cohort_root_path)
    mask_pixel_standaradization('./fat/cropped_mask')