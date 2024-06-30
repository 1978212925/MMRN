# encoding=utf-8
import os
import sys

import torchvision
from torch import nn

from utils.data_transform import node_transform
from utils.get_dataset import getdataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
# from models.swin_transfomer import swin_small_patch4_window7_224
import os
import sys
from utils import get_five_fold_dataset
from models.cla_models import swin_small_patch4_window7_224
from models.cla_models.resnet_OOD import ResModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def cal_mean_CI(list):
    # 计算均值
    mean = np.mean(list)
    # 十折交叉验证所以自由度是10
    n = 5
    # 计算标准差
    std = np.std(list)
    # 双边alpha0.05 = 2.228
    a = 2.571

    return mean, [mean - (a * std / np.sqrt(n)), mean + (a * std / np.sqrt(n))]


def cal_acc_sens_spec_ppv_npv(threshold, y_true, y_pred):
    for j in range(len(y_pred)):
        if y_pred[j] > threshold:
            y_pred[j] = 1
        else:
            y_pred[j] = 0
    y_pred_argmax = y_pred
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_pred_argmax)):
        if y_pred_argmax[i] == y_true[i]:
            if y_true[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_true[i] == 0:
                fp += 1
            else:
                fn += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn + 0.00001)
    spec = tn / (tn + fp + 0.00001)
    ppv = tp / (tp + fp + 0.00001)
    npv = tn / (tn + fn + 0.00001)
    return acc, sens, spec, ppv, npv


'''
        'ResModelOOD_node 第1折 0.8488 0.7206 0.7751 0.6349.pth',
        'ResModelOOD_node 第2折 0.8279 0.7206 0.7761 0.6061.pth',
        'ResModelOOD_node 第3折 0.8317 0.7081 0.7485 0.6289.pth',
        'ResModelOOD_node 第4折 0.7457 0.7087 0.7811 0.6253.pth',
        'ResModelOOD_node 第5折 0.8065 0.7575 0.7414 0.6183.pth'
'''

'''
'VGG_node 第1折 0.7924 0.7326.pth',
'VGG_node 第2折 0.7929 0.7889.pth',
'VGG_node 第3折 0.7696 0.7429.pth',
'VGG_node 第4折 0.7799 0.7750.pth',
'VGG_node 第5折 0.7819 0.7321.pth'
'''

'''
'ResNet_node 第1折 0.8097 0.7317 0.7239 0.6406.pth',
'ResNet_node 第2折 0.8366 0.7746 0.7872 0.6231.pth',
'ResNet_node 第3折 0.7845 0.7219 0.7589 0.5972.pth',
'ResNet_node 第4折 0.7966 0.8065 0.7990 0.6297.pth',
'ResNet_node 第5折 0.8150 0.8062 0.7350 0.6234.pth'
'''

'''
'ResNet50_node 第1折 0.7317 0.7032 0.7407 0.6435.pth',
'ResNet50_node 第2折 0.7406 0.7336 0.7562 0.6584.pth',
'ResNet50_node 第3折 0.7690 0.7236 0.7448 0.6416.pth',
'ResNet50_node 第4折 0.7416 0.7132 0.7522 0.6587.pth',
'ResNet50_node 第5折 0.7450 0.7715 0.7694 0.6229.pth'
'''

'''
'DenseNet121_node 第1折 0.7936 0.7525 0.7707 0.6202.pth',
'DenseNet121_node 第2折 0.7757 0.7448 0.7872 0.6272.pth',
'DenseNet121_node 第3折 0.7729 0.7268 0.7508 0.6387.pth',
'DenseNet121_node 第4折 0.7690 0.7429 0.7650 0.6400.pth',
'DenseNet121_node 第5折 0.7594 0.7966 0.7667 0.6345.pth'
'''

'''
'DenseNet161_node 第1折 0.7605 0.7349 0.7613 0.6285.pth',
'DenseNet161_node 第2折 0.7645 0.7548 0.7606 0.6380.pth',
'DenseNet161_node 第3折 0.7729 0.7383 0.7505 0.6437.pth',
'DenseNet161_node 第4折 0.7492 0.7353 0.7549 0.6301.pth',
'DenseNet161_node 第5折 0.7547 0.7905 0.7778 0.6296.pth'
'''

'''
'ResNet101_node 第1折 0.8033 0.7211 0.7364 0.6233.pth',
'ResNet101_node 第2折 0.7352 0.7253 0.7532 0.6402.pth',
'ResNet101_node 第3折 0.7561 0.6981 0.7545 0.6462.pth',
'ResNet101_node 第4折 0.7441 0.7253 0.7667 0.6401 (1).pth',
'ResNet101_node 第4折 0.7441 0.7253 0.7667 0.6401 (1).pth'
'''
# 分别输出每一折对应的训练集和验证集的auc
if __name__ == '__main__':
    # 每一折保存的模型名称
    model_name = 'ResNet34'
    fold_model_dict = [
        'ResNet_node 第1折 0.8097 0.7317 0.7239 0.6406.pth',
'ResNet_node 第2折 0.8366 0.7746 0.7872 0.6231.pth',
'ResNet_node 第3折 0.7845 0.7219 0.7589 0.5972.pth',
'ResNet_node 第4折 0.7966 0.8065 0.7990 0.6297.pth',
'ResNet_node 第5折 0.8150 0.8062 0.7350 0.6234.pth'
    ]
    test_name_list = ['internal_test_cohort', 'external_test_cohort1']
    for test_name in test_name_list:
        test_dataset = getdataset("../dataset/thyroid_cancer_LM.csv", f"../dataset/cropped/{test_name}", "",
                                  node_transform['train'], mode_select=3, is_augment=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12)
        all_fold_test_auc = []
        all_fold_test_acc = []
        all_fold_test_sens = []
        all_fold_test_spec = []
        all_fold_test_ppv = []
        all_fold_test_npv = []
        print(f'验证集{test_name}结果')
        # i 控制折数
        for i in range(len(fold_model_dict)):
            # 每一折创建一个新的模型，并加载该模型对应的参数
            model_dict_path = os.path.join(f"../runner/Model_Dict/{model_name}", fold_model_dict[i])
            # 模型
            # model = torchvision.models.vgg19()
            # num_features = model.classifier[-1].in_features
            # model.classifier[-1] = nn.Linear(num_features, 2)

            model = torchvision.models.resnet34()
            model.fc = nn.Linear(512, 2, bias=True)

            # model = torchvision.models.resnet50()
            # model.fc = nn.Linear(2048, 2, bias=True)

            # model = torchvision.models.resnet101()
            # model.fc = nn.Linear(2048, 2, bias=True)

            # model = torchvision.models.densenet121()
            # model.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)

            # model = torchvision.models.densenet161()
            # model.classifier = nn.Linear(in_features=2208, out_features=2, bias=True)

            # model = ResModel(2)
            model.load_state_dict(torch.load(model_dict_path))
            model.cuda()
            model.eval()

            test_batch_true = []
            test_batch_pred = []

            # 计算训练集每一个batch的模型预测值，并保存
            for data in tqdm(test_dataloader):
                img, label = data
                img = img.to('cuda:0')
                pred = model(img).softmax(dim=1)[:, 1]
                pred = pred.cpu().detach().numpy()
                test_batch_pred.append(pred)
                test_batch_true.append(label)

            test_y_pred = np.concatenate(test_batch_pred)
            test_y_true = np.concatenate(test_batch_true)

            fpr, tpr, thresholds = roc_curve(test_y_true, test_y_pred)
            best_threshold = thresholds[np.argmax(tpr - fpr)]
            fold_roc_auc = auc(fpr, tpr)
            fold_acc, fold_sens, fold_spec, fold_ppv, fold_npv = cal_acc_sens_spec_ppv_npv(best_threshold, test_y_true,
                                                                                           test_y_pred)
            print(
                f"第{i + 1}折测试集AUC:{fold_roc_auc}, ACC:{fold_acc}, SENS:{fold_sens}, SPEC{fold_spec}:, PPV:{fold_ppv}, PPV:{fold_npv}")
            all_fold_test_auc.append(fold_roc_auc)
            all_fold_test_acc.append(fold_acc)
            all_fold_test_sens.append(fold_sens)
            all_fold_test_spec.append(fold_spec)
            all_fold_test_ppv.append(fold_ppv)
            all_fold_test_npv.append(fold_npv)

            # 计算最终的平均值，以及置信区间
        auc_mean, auc_ci = cal_mean_CI(all_fold_test_auc)
        acc_mean, acc_ci = cal_mean_CI(all_fold_test_acc)
        sens_mean, sens_ci = cal_mean_CI(all_fold_test_sens)
        spec_mean, spec_ci = cal_mean_CI(all_fold_test_spec)
        ppv_mean, ppv_ci = cal_mean_CI(all_fold_test_ppv)
        npv_mean, npv_ci = cal_mean_CI(all_fold_test_npv)

        print('AUC: {:.3f} [{:.3f}, {:.3f}]'.format(auc_mean, auc_ci[0], auc_ci[1]))
        print('ACC: {:.3f} [{:.3f}, {:.3f}]'.format(acc_mean, acc_ci[0], acc_ci[1]))
        print('SENS: {:.3f} [{:.3f}, {:.3f}]'.format(sens_mean, sens_ci[0], sens_ci[1]))
        print('SPEC: {:.3f} [{:.3f}, {:.3f}]'.format(spec_mean, spec_ci[0], spec_ci[1]))
        print('PPV: {:.3f} [{:.3f}, {:.3f}]'.format(ppv_mean, ppv_ci[0], ppv_ci[1]))
        print('NPV: {:.3f} [{:.3f}, {:.3f}]'.format(npv_mean, npv_ci[0], npv_ci[1]))
