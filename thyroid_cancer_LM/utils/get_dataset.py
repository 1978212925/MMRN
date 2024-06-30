import os
import sys

from torchvision.transforms import transforms

from utils.data_transform import node_transform
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import PIL.Image
import pandas as pd
import torch.utils.data
from torch.utils.data.dataset import Dataset
from utils import data_transform
import cv2
import numpy as np

from torchvision.transforms import functional as F
def get_csv_data(csv_path):
    df = pd.read_csv(csv_path, encoding='GBK', usecols=[0, 1])
    # df.drop([0], inplace=True)
    label = df.to_numpy()
    return label


class CancerDataset(Dataset):
    def __init__(self, img_content_path_list, label_list, fat_content_path_list, transform, mode_select=0, augment=False):
        self.img_content_path_list = img_content_path_list
        self.label_list = label_list
        self.transform = transform
        self.mode_select = mode_select
        self.fat_content_path_list = fat_content_path_list
        self.fat_content_path_list_len = len(fat_content_path_list)
        self.augment = augment
    def __getitem__(self, index):
        # 拿到裁剪图像路径dataset/cropped/primary_cohort/697106.jpg
        img_content_path = self.img_content_path_list[index]
        merge_content_path = img_content_path.replace('cropped', 'merge_region')
        peritumor_content_path = merge_content_path.replace("merge_region", "peritumor")
        intratumor_content_path = merge_content_path.replace("merge_region", "intratumor")
        label = self.label_list[index]
        # 如果没传入脂肪路径，但是模式选择为5了，报错
        if self.fat_content_path_list_len == 0 and self.mode_select == 5:
            print('mode select is 5 but not input fat path')
            return 1
        elif self.fat_content_path_list_len == 0:
            # 只使用裁剪后的原图像
            if self.mode_select == 0:
                cropped_img = cv2.imread(img_content_path)
                cropped_img = PIL.Image.fromarray(cropped_img)
                cropped_img = self.transform(cropped_img)
                data = cropped_img
            # 只使用瘤周图像
            elif self.mode_select == 1:
                peritumor = cv2.imread(peritumor_content_path)
                peritumor = PIL.Image.fromarray(peritumor)
                peritumor = self.transform(peritumor)
                data = peritumor
            # 只使用瘤内图像
            elif self.mode_select == 2:
                intratumor = cv2.imread(intratumor_content_path)
                intratumor = PIL.Image.fromarray(intratumor)
                intratumor = self.transform(intratumor)
                data = intratumor
            # 只使用瘤周瘤内混合图像
            elif self.mode_select == 3:
                merge = cv2.imread(merge_content_path)
                merge = PIL.Image.fromarray(merge)
                merge = self.transform(merge)
                data = merge
            # 瘤周瘤内混合三图像合并
            elif self.mode_select == 4:
                merge = cv2.imread(peritumor_content_path, flags=cv2.IMREAD_GRAYSCALE)
                peritumor = cv2.imread(intratumor_content_path, flags=cv2.IMREAD_GRAYSCALE)
                intratumor = cv2.imread(merge_content_path, flags=cv2.IMREAD_GRAYSCALE)
                # cv2.imshow("1", peritumor)
                # cv2.imshow("2", intratumor)
                # cv2.imshow("3", merge)
                # cv2.waitKey(-1)
                merge = np.expand_dims(merge, 2)
                peritumor = np.expand_dims(peritumor, 2)
                intratumor = np.expand_dims(intratumor, 2)

                concat_img = np.concatenate((merge, peritumor, intratumor), 2)
                # cv2.imshow("4", data)
                # cv2.waitKey(-1)
                concat_img = PIL.Image.fromarray(concat_img)
                concat_img = self.transform(concat_img)
                data = concat_img
            else:
                print("mode select error, must in range (0-5)")
                data = None
            return data, label
        else:
            # 拿到脂肪图像路径
            fat_content_path = self.fat_content_path_list[index]
            # 只使用裁剪后的原图像
            if self.mode_select == 0:
                cropped_img = cv2.imread(img_content_path)
                cropped_img = PIL.Image.fromarray(cropped_img)
                cropped_img = self.transform(cropped_img)
                data = cropped_img
            # 只使用瘤周图像
            elif self.mode_select == 1:
                peritumor = cv2.imread(peritumor_content_path)
                peritumor = PIL.Image.fromarray(peritumor)
                peritumor = self.transform(peritumor)
                data = peritumor
            # 只使用瘤内图像
            elif self.mode_select == 2:
                intratumor = cv2.imread(intratumor_content_path)
                intratumor = PIL.Image.fromarray(intratumor)
                intratumor = self.transform(intratumor)
                data = intratumor
            # 只使用瘤周瘤内混合图像
            elif self.mode_select == 3:
                merge = cv2.imread(merge_content_path)
                merge = PIL.Image.fromarray(merge)
                # if self.augment:
                #     merge = F.adjust_brightness(merge, 2)
                merge = self.transform(merge)
                data = merge
            # 瘤周瘤内混合三图像合并
            elif self.mode_select == 4:
                merge = cv2.imread(peritumor_content_path, flags=cv2.IMREAD_GRAYSCALE)
                peritumor = cv2.imread(intratumor_content_path, flags=cv2.IMREAD_GRAYSCALE)
                intratumor = cv2.imread(merge_content_path, flags=cv2.IMREAD_GRAYSCALE)
                # cv2.imshow("1", peritumor)
                # cv2.imshow("2", intratumor)
                # cv2.imshow("3", merge)
                # cv2.waitKey(-1)
                merge = np.expand_dims(merge, 2)
                peritumor = np.expand_dims(peritumor, 2)
                intratumor = np.expand_dims(intratumor, 2)

                concat_img = np.concatenate((merge, peritumor, intratumor), 2)
                # cv2.imshow("4", data)
                # cv2.waitKey(-1)
                concat_img = PIL.Image.fromarray(concat_img)
                concat_img = self.transform(concat_img)
                data = concat_img
            # 只使用脂肪图像
            elif self.mode_select == 5:
                fat_img = cv2.imread(fat_content_path, flags=cv2.IMREAD_COLOR)
                fat_img = fat_img
                fat_img = PIL.Image.fromarray(fat_img)
                fat_img = self.transform(fat_img)
                data = fat_img
            else:
                print("mode select error, must in range (0-5)")
                data = None
            return data, label


    def __len__(self):
        return len(self.img_content_path_list)


def getdataset(csv_path, node_path, fat_path, transform, mode_select=0, is_augment=False):
    label_numpy = get_csv_data(csv_path)
    img_name_list = os.listdir(node_path)
    img_num = label_numpy[:, 0]
    img_num = img_num.astype(np.int64)
    img_label = label_numpy[:, 1].astype(int)
    img_num, img_label = img_num.tolist(), img_label.tolist()
    new_img_content_path_list = []
    new_label_list = []
    new_fat_content_path_list = []

    for img_name in img_name_list:
        img_name_int = int(img_name.replace(".jpg", ""))
        try:
            img_name_index = img_num.index(img_name_int)
            new_label_list.append(img_label[img_name_index])
            new_img_content_path_list.append(os.path.join(node_path, img_name))
            if len(fat_path) != 0:
                new_fat_content_path_list.append(os.path.join(fat_path, img_name.replace('.jpg', '.png')))
            else:
                pass
        except Exception as err:
            pass

    print(len(new_img_content_path_list), len(new_label_list), len(new_fat_content_path_list))
    if is_augment:
        aug_transform = data_transform.aug_transform
        totaldataset1 = CancerDataset(new_img_content_path_list, new_label_list, new_fat_content_path_list, transform,
                                    mode_select=mode_select)
        # only_0_img_list = []
        # only_0_label_list = []
        # only_0_fat_list = []
        # only_0_fat_img_list = []
        # for i in range(len(new_label_list)):
        #     if new_label_list[i] == 0:
        #         only_0_img_list.append(new_img_list[i])
        #         only_0_label_list.append(new_label_list[i])
        #         # only_0_fat_list.append(new_fat_list[i])
        #         only_0_fat_img_list.append(new_fat_img_list[i])
        totaldataset2 = CancerDataset(new_img_content_path_list, new_label_list, new_fat_content_path_list, aug_transform,
                                    mode_select=mode_select, augment=True)
        totaldataset = totaldataset1+totaldataset2
    else:
        totaldataset = CancerDataset(new_img_content_path_list, new_label_list, new_fat_content_path_list, transform, mode_select=mode_select)
    return totaldataset


if __name__ == '__main__':
    total_dataset = getdataset("../dataset/thyroid_cancer_LM.csv", "../dataset/cropped/primary_cohort", "../dataset/fat",
                               node_transform['train'], mode_select=5)
    print(total_dataset[30])
