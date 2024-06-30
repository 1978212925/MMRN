# encoding=utf-8
import copy
import os

import pandas as pd
import torch
import torchvision
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

from models.cla_models import ResModel
from utils import get_five_fold_dataset
from utils.data_transform import node_transform
from utils.get_dataset import getdataset

from utils.get_five_fold_dataset import get_five_fold_dataloaders

train_val_dataloader = get_five_fold_dataloaders(csv_path="../../dataset/thyroid_cancer_LM.csv", node_path="../../dataset/cropped/primary_cohort",
                             fat_path="../../dataset/fat", transform_mode='train', mode_select=3)
fold_model_dict = [
        '../../runner/Model_Dict/Resnet50/ResNet50_node 第1折 0.7317 0.7032 0.7407 0.6435.pth',
        '../../runner/Model_Dict/Resnet50/ResNet50_node 第2折 0.7406 0.7336 0.7562 0.6584.pth',
        '../../runner/Model_Dict/Resnet50/ResNet50_node 第3折 0.7690 0.7236 0.7448 0.6416.pth',
        '../../runner/Model_Dict/Resnet50/ResNet50_node 第4折 0.7416 0.7132 0.7522 0.6587.pth',
        '../../runner/Model_Dict/Resnet50/ResNet50_node 第5折 0.7450 0.7715 0.7694 0.6229.pth'
]

# 第n折
for fold, (train_loader, val_loader) in enumerate(train_val_dataloader):
    if True:
        test0_dataset = getdataset("../../dataset/thyroid_cancer_LM.csv", "../../dataset/cropped/internal_test_cohort", "../../dataset/fat",
                                    node_transform['train'], mode_select=3, is_augment=False)
        test1_dataset = getdataset("../../dataset/thyroid_cancer_LM.csv", "../../dataset/cropped/external_test_cohort1", "../../dataset/fat",
                                    node_transform['train'], mode_select=3, is_augment=False)

        test0_name_list = copy.deepcopy(test0_dataset.img_content_path_list)
        test1_name_list = copy.deepcopy(test1_dataset.img_content_path_list)

        test0_dataloader = torch.utils.data.DataLoader(test0_dataset, batch_size=1)
        test1_dataloader = torch.utils.data.DataLoader(test1_dataset, batch_size=1)

        train_name_list = []
        val_name_list = []
        train_name_list_all = train_loader.dataset.datasets[0].img_content_path_list
        val_name_list_all = val_loader.dataset.datasets[0].img_content_path_list
        train_name_list_all += train_name_list_all[:]
        val_name_list_all += val_name_list_all[:]
        # 用来存储图像id号
        for sam in train_loader.sampler:
            train_name_list.append(train_name_list_all[sam])
        for sam2 in val_loader.sampler:
            val_name_list.append(val_name_list_all[sam2])

        all_dataloader = [train_loader, val_loader, test0_dataloader, test1_dataloader]
        all_data_name = [train_name_list, val_name_list, test0_name_list, test1_name_list]
        all_csv_name = ['train_DL_FC_result', 'val_DL_FC_result', 'test0_DL_FC_result', 'test1_DL_FC_result']
        model = torchvision.models.resnet50()
        model.fc = nn.Linear(2048, 2, bias=True)
        model.load_state_dict(torch.load(fold_model_dict[fold]))
        model.cuda()
        model.eval()
        # 第n折中每个cohort
        for i in range(len(all_dataloader)):
            # 标签，全连接层结果，预测值
            name_list = all_data_name[i]
            for name_index in range(len(name_list)):
                name_split = name_list[name_index].split('\\')
                name_list[name_index] = name_split[-1].replace('.jpg', '')
            cohort_label_list = []
            cohort_y_pred_list = []
            for data in tqdm(all_dataloader[i]):
                img, label = data
                img = img.to('cuda:0')
                pred = model(img)
                pred = pred.softmax(dim=1)[:, 1]
                pred = pred.cpu().detach().numpy()
                cohort_label_list.append(label)
                cohort_y_pred_list.append(pred)


            id = [str(sublist) for sublist in name_list]
            label = [int(sublist) for sublist in cohort_label_list]
            y_pred = [float(sublist) for sublist in cohort_y_pred_list]
            auc = roc_auc_score(label, y_pred)
            print(f'Fold: {fold}, {all_csv_name[i]} AUC: {auc}')
            final_data = {'ID': id, 'Label': label, 'Y_pred': y_pred}

            df = pd.DataFrame(final_data)
            df.to_csv(f'node_fold_data/fold_{fold + 1}/{all_csv_name[i]}_{fold + 1}.csv', index=False)

