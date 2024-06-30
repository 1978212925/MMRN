import os
import sys

import torchvision
from torch import nn


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

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
def cal_mean_CI(list):
    # 计算均值
    mean = np.mean(list)
    # 十折交叉验证所以自由度是10
    n = 5
    # 计算标准差
    std = np.std(list)
    # 双边alpha0.05 = 2.571
    a = 2.571

    return mean, [mean - (a * std/np.sqrt(n)), mean + (a * std/np.sqrt(n))]
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
'DenseNet_node 第1折 0.8749 0.7795.pth',
'DenseNet_node 第2折 0.7859 0.7784.pth',
'DenseNet_node 第3折 0.8574 0.7159.pth',
'DenseNet_node 第4折 0.8161 0.7025.pth',
'DenseNet_node 第5折 0.8399 0.7575.pth'
'''

'''
'VGG_node 第1折 0.7924 0.7326.pth',
'VGG_node 第2折 0.7929 0.7889.pth',
'VGG_node 第3折 0.7696 0.7429.pth',
'VGG_node 第4折 0.7799 0.7750.pth',
'VGG_node 第5折 0.7819 0.7321.pth'
'''
if __name__ == '__main__':
    model_name = 'Resnet50'
    # 每一折保存的模型名称
    fold_model_dict = [
        'ResNet50_node 第1折 0.7317 0.7032 0.7407 0.6435.pth',
        'ResNet50_node 第2折 0.7406 0.7336 0.7562 0.6584.pth',
        'ResNet50_node 第3折 0.7690 0.7236 0.7448 0.6416.pth',
        'ResNet50_node 第4折 0.7416 0.7132 0.7522 0.6587.pth',
        'ResNet50_node 第5折 0.7450 0.7715 0.7694 0.6229.pth'
                       ]
    all_fold_dataloader = get_five_fold_dataset.get_five_fold_dataloaders(transform_mode='train', mode_select=3)
    all_fold_train_auc = []
    all_fold_train_acc = []
    all_fold_train_sens = []
    all_fold_train_spec = []
    all_fold_train_ppv = []
    all_fold_train_npv = []

    all_fold_val_auc = []
    all_fold_val_acc = []
    all_fold_val_sens = []
    all_fold_val_spec = []
    all_fold_val_ppv = []
    all_fold_val_npv = []

    # i 控制折数
    for i in range(len(all_fold_dataloader)):
        # 每一折创建一个新的模型，并加载该模型对应的参数
        model_dict_path = os.path.join(f"../runner/Model_Dict/{model_name}", fold_model_dict[i])
        # 模型
        # model = torchvision.models.vgg19()
        # num_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(num_features, 2)

        # model = torchvision.models.resnet34()
        # model.fc = nn.Linear(512, 2, bias=True)

        model = torchvision.models.resnet50()
        model.fc = nn.Linear(2048, 2, bias=True)

        # model = torchvision.models.resnet101()
        # model.fc = nn.Linear(2048, 2, bias=True)

        # model = torchvision.models.densenet121()
        # model.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)

        # model = torchvision.models.densenet161()
        # model.classifier = nn.Linear(in_features=2208, out_features=2, bias=True)

        model.load_state_dict(torch.load(model_dict_path))
        model.cuda()
        model.eval()

        fold_dataloader = all_fold_dataloader[i]
        train_loader, val_loader = fold_dataloader
        train_batch_true = []
        train_batch_pred = []
        val_batch_true = []
        val_batch_pred = []
        # 计算训练集每一个batch的模型预测值，并保存
        for data in tqdm(train_loader):
            img, label = data
            img = img.to('cuda:0')
            pred = model(img).softmax(dim=1)[:, 1]
            pred = pred.cpu().detach().numpy()
            train_batch_pred.append(pred)
            train_batch_true.append(label)

        # 计算验证集每一个batch的模型预测值，并保存
        for data in tqdm(val_loader):
            img, label = data
            img = img.to('cuda:0')
            pred = model(img).softmax(dim=1)[:, 1]
            pred = pred.cpu().detach().numpy()
            val_batch_pred.append(pred)
            val_batch_true.append(label)

        train_y_pred = np.concatenate(train_batch_pred)
        train_y_true = np.concatenate(train_batch_true)

        fpr, tpr, thresholds = roc_curve(train_y_true, train_y_pred)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        fold_roc_auc = auc(fpr, tpr)
        fold_acc, fold_sens, fold_spec, fold_ppv, fold_npv = cal_acc_sens_spec_ppv_npv(best_threshold, train_y_true,
                                                                                       train_y_pred)
        print(f"第{i + 1}折训练集AUC:{fold_roc_auc}, ACC:{fold_acc}, SENS:{fold_sens}, SPEC{fold_spec}:, PPV:{fold_ppv}, PPV:{fold_npv}")
        all_fold_train_auc.append(fold_roc_auc)
        all_fold_train_acc.append(fold_acc)
        all_fold_train_sens.append(fold_sens)
        all_fold_train_spec.append(fold_spec)
        all_fold_train_ppv.append(fold_ppv)
        all_fold_train_npv.append(fold_npv)

        val_y_pred = np.concatenate(val_batch_pred)
        val_y_true = np.concatenate(val_batch_true)
        fpr, tpr, thresholds = roc_curve(val_y_true, val_y_pred)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        fold_roc_auc = auc(fpr, tpr)
        fold_acc, fold_sens, fold_spec, fold_ppv, fold_npv = cal_acc_sens_spec_ppv_npv(best_threshold, val_y_true,
                                                                                       val_y_pred)
        print(f"第{i + 1}折验证集AUC:{fold_roc_auc}, ACC:{fold_acc}, SENS:{fold_sens}, SPEC{fold_spec}:, PPV:{fold_ppv}, PPV:{fold_npv}")
        all_fold_val_auc.append(fold_roc_auc)
        all_fold_val_acc.append(fold_acc)
        all_fold_val_sens.append(fold_sens)
        all_fold_val_spec.append(fold_spec)
        all_fold_val_ppv.append(fold_ppv)
        all_fold_val_npv.append(fold_npv)

    auc_mean, auc_ci = cal_mean_CI(all_fold_train_auc)
    acc_mean, acc_ci = cal_mean_CI(all_fold_train_acc)
    sens_mean, sens_ci = cal_mean_CI(all_fold_train_sens)
    spec_mean, spec_ci = cal_mean_CI(all_fold_train_spec)
    ppv_mean, ppv_ci = cal_mean_CI(all_fold_train_ppv)
    npv_mean, npv_ci = cal_mean_CI(all_fold_train_npv)
    print('训练集最终的结果是:')
    print(f"AUC:{auc_mean}, {auc_ci}")
    print(f"ACC:{acc_mean}, {acc_ci}")
    print(f"SENS:{sens_mean}, {sens_ci}")
    print(f"SPEC:{spec_mean}, {spec_ci}")
    print(f"PPV:{ppv_mean}, {ppv_ci}")
    print(f"NPV:{npv_mean}, {npv_ci}")

    auc_mean, auc_ci = cal_mean_CI(all_fold_val_auc)
    acc_mean, acc_ci = cal_mean_CI(all_fold_val_acc)
    sens_mean, sens_ci = cal_mean_CI(all_fold_val_sens)
    spec_mean, spec_ci = cal_mean_CI(all_fold_val_spec)
    ppv_mean, ppv_ci = cal_mean_CI(all_fold_val_ppv)
    npv_mean, npv_ci = cal_mean_CI(all_fold_val_npv)
    print('验证集最终的结果是:')
    print(f"AUC:{auc_mean}, {auc_ci}")
    print(f"ACC:{acc_mean}, {acc_ci}")
    print(f"SENS:{sens_mean}, {sens_ci}")
    print(f"SPEC:{spec_mean}, {spec_ci}")
    print(f"PPV:{ppv_mean}, {ppv_ci}")
    print(f"NPV:{npv_mean}, {npv_ci}")