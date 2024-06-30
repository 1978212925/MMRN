# encoding=utf-8
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
plt.rc('font', family='Times New Roman')

all_csv_name = ['test0', 'test1']
all_csv_result = [[0.802,0.796,0.807,0.758,0.746,0.770,0.628,0.627,0.629,0.803,0.798,0.809,0.829,0.822,0.835],
                  [0.760,0.756,0.764,0.759,0.742,0.776,0.560,0.559,0.561,0.795,0.782,0.808,0.818,0.808,0.828]]
for i in range(len(all_csv_name)):
    csv_label = []
    csv_Clinical = []
    csv_IPN = []
    csv_Adipose = []
    csv_Clinical_IPN = []
    csv_DLRN = []
    cohort_result = all_csv_result[i]
    for fold in range(1, 6):
        data = pd.read_csv(f'../../utils/nomogram/dif_features_combin_result/fold_{fold}/{all_csv_name[i]}_dif_features_com_result_{fold}.csv')
        y_true = np.array(data.iloc[:, 1]).astype(int)
        # Clinical
        y_pred1 = np.array(data.iloc[:, 2]).astype(float)
        # IPN
        y_pred2 = np.array(data.iloc[:, 3]).astype(float)
        # Adipose_radiomic
        y_pred3 = np.array(data.iloc[:, 4]).astype(float)
        # Clinical_IPN
        y_pred4 = np.array(data.iloc[:, 6]).astype(float)
        # DLRN
        y_pred5 = np.array(data.iloc[:, 7]).astype(float)

        csv_label.extend(y_true)
        csv_Clinical.extend(y_pred1)
        csv_IPN.extend(y_pred2)
        csv_Adipose.extend(y_pred3)
        csv_Clinical_IPN.extend(y_pred4)
        csv_DLRN.extend(y_pred5)

    fpr1, tpr1, thresholds1 = roc_curve(csv_label, csv_Clinical)
    fpr2, tpr2, thresholds2 = roc_curve(csv_label, csv_IPN)
    fpr3, tpr3, thresholds3 = roc_curve(csv_label, csv_Adipose)
    fpr4, tpr4, thresholds4 = roc_curve(csv_label, csv_Clinical_IPN)
    fpr5, tpr5, thresholds5 = roc_curve(csv_label, csv_DLRN)
    print(auc(fpr1, tpr1), auc(fpr2, tpr2), auc(fpr3, tpr3), auc(fpr4, tpr4), auc(fpr5, tpr5))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr1, tpr1, linewidth=2.5, label='Clinical model\nAUC {:.3f} (95% CI: {:.3f}–{:.3f})'.format(cohort_result[0],cohort_result[1],cohort_result[2]))
    ax.plot(fpr2, tpr2, linewidth=2.5, label='PTC signature\nAUC {:.3f} (95% CI: {:.3f}–{:.3f}))'.format(cohort_result[3],cohort_result[4],cohort_result[5]))
    ax.plot(fpr3, tpr3, linewidth=2.5, label='Fat model\nAUC {:.3f} (95% CI: {:.3f}–{:.3f})'.format(cohort_result[6],cohort_result[7],cohort_result[8]))
    ax.plot(fpr4, tpr4, linewidth=2.5, label='Clinical PTC model\nAUC {:.3f} (95% CI: {:.3f}–{:.3f})'.format(cohort_result[9],cohort_result[10],cohort_result[11]))
    ax.plot(fpr5, tpr5, linewidth=2.5, label='MMRN\nAUC {:.3f} (95% CI: {:.3f}–{:.3f})'.format(cohort_result[12],cohort_result[13],cohort_result[14]))
    ax.set_xlabel('1-Specificity', fontsize=15)
    ax.set_ylabel('Sensitivity', fontsize=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # ax.set_title('ROC comparision')
    ax.legend(fontsize=15)
    plt.tight_layout()  # 调整布局
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.savefig(f'./img/roc_curves_nomogram_{all_csv_name[i]}.png', dpi=1000)  # 保存为文件
    plt.show()


