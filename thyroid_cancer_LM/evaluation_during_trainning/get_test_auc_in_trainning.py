import math
import scipy
import torch
import numpy as np
import torchvision
from scipy.stats import sem, t, stats
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from utils.get_dataset import getdataset
# from models.swin_transfomer import swin_small_patch4_window7_224
from utils.data_transform import node_transform
import os
import sys
from models.cla_models import swin_small_patch4_window7_224

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if __name__ == '__main__':
    # mode_select = int(input("please select data fuse mode(range is 0-5)："))
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    # 加载模型参数
    mode_select = 4
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
    model.load_state_dict(torch.load('../runner/Running_Dict/ResNet_node 第1折 0.7989 0.8260.pth'))
    model.eval()
    model.cuda()
    data_transform = node_transform
    # # 加载测试数据集
    total_dataset = getdataset("../dataset/thyroid_cancer_LM.csv", "../dataset/cropped/internal_test_cohort", "",
                               node_transform['train'], mode_select=mode_select, is_augment=False)
    total_loader = torch.utils.data.DataLoader(total_dataset, batch_size=32)
    imgs, labels = next(iter(total_loader))
    # plt.figure(figsize=(16, 8))
    # for i in range(len(imgs[:8])):
    #     img = imgs[:8][i]
    #     lable = labels[:8][i]
    #     img = img.numpy()
    #     img = np.transpose(img, (1, 2, 0))
    #     plt.subplot(2, 4, i + 1)
    #     plt.imshow(img)
    #     plt.title(lable)
    # plt.show()
    print(len(total_loader.dataset))

    # 输出测试结果
    y_true = []
    y_score = []
    with torch.no_grad():
        for data, labels in tqdm(total_loader):
            imgs = data
            imgs = imgs.to('cuda:0')
            # 预测概率
            y_pred = model(imgs).softmax(dim=1)[:, 1]
            y_true.append(labels.numpy())
            y_score.append(y_pred.cpu().numpy())
    # 把每个 batch 的结果合并成一个数组
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print("ROC is {:.3f}".format(roc_auc))
    # 找到最佳阈值
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    print(best_threshold)
    # 计算spe，sens 阈值越小spe越高，阈值越大sens越高
    for j in range(len(y_score)):
        if y_score[j] > 0.5:
            y_score[j] = 1
        else:
            y_score[j] = 0
    y_pred_argmax = y_score
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
    epoch_sens = tp / (tp + fn + 0.00001)
    epoch_spe = tn / (tn + fp + 0.00001)
    epoch_acc = (tp+tn)/(tp+tn+fp+fn)
    print("sens={:.3f}, spe={:.3f}".format(epoch_sens, epoch_spe))
    print("acc={:.3f}".format(epoch_acc))