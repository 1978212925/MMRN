# encoding=utf-8
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

plt.rc('font', family='Times New Roman')

all_csv_name = ['test0', 'test1']


def cal_sens(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_true[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_true[i] == 0:
                fp += 1
            else:
                fn += 1

    sens = tp / (tp + fn + 0.00001)
    return sens


def cal_spec(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_true[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_true[i] == 0:
                fp += 1
            else:
                fn += 1
    spec = tn / (tn + fp + 0.00001)
    return spec


def cal_CI(list):
    # 计算均值
    mean = np.mean(list)
    # 5折交叉验证所以自由度是5
    n = 5
    # 计算标准差
    std = np.std(list)
    # 双边alpha0.05 = 2.571
    a = 2.571
    return mean - (a * std / np.sqrt(n)), mean + (a * std / np.sqrt(n))


all_csv_label = []
all_csv_DLRN = []
all_csv_result = [[0.829, 0.822, 0.835],
                  [0.818, 0.808, 0.828]]
all_csv_sens_mean = []
all_csv_sens_CI = []
all_csv_fpr_mean = []
all_csv_fpr_CI = []
for i in range(len(all_csv_name)):
    csv_label = []
    csv_DLRN = []
    for fold in range(1, 6):
        data = pd.read_csv(
            f'../../utils/nomogram/dif_features_combin_result/fold_{fold}/{all_csv_name[i]}_dif_features_com_result_{fold}.csv')
        y_true = np.array(data.iloc[:, 1]).astype(int)
        # DLRN
        y_pred5 = np.array(data.iloc[:, 7]).astype(float)

        csv_label.extend(y_true)
        csv_DLRN.extend(y_pred5)

    all_csv_label.append(csv_label)
    all_csv_DLRN.append(csv_DLRN)
    physician1 = pd.read_csv(f'../../utils/physician/physician_result/physician1/{all_csv_name[i]}_final.csv')
    physician2 = pd.read_csv(f'../../utils/physician/physician_result/physician2/{all_csv_name[i]}_final.csv')
    physician3 = pd.read_csv(f'../../utils/physician/physician_result/physician3/{all_csv_name[i]}_final.csv')
    physician4 = pd.read_csv(f'../../utils/physician/physician_result/physician4/{all_csv_name[i]}_final.csv')
    physician5 = pd.read_csv(f'../../utils/physician/physician_result/physician5/{all_csv_name[i]}_final.csv')

    physician1_label = np.array(physician1.iloc[:, 1]).astype(int)
    physician1_pred = np.array(physician1.iloc[:, 2]).astype(int)
    physician2_label = np.array(physician2.iloc[:, 1]).astype(int)
    physician2_pred = np.array(physician2.iloc[:, 2]).astype(int)
    physician3_label = np.array(physician3.iloc[:, 1]).astype(int)
    physician3_pred = np.array(physician3.iloc[:, 2]).astype(int)
    physician4_label = np.array(physician4.iloc[:, 1]).astype(int)
    physician4_pred = np.array(physician4.iloc[:, 2]).astype(int)
    physician5_label = np.array(physician5.iloc[:, 1]).astype(int)
    physician5_pred = np.array(physician5.iloc[:, 2]).astype(int)

    physician1_sens = cal_sens(physician1_label, physician1_pred)
    physician1_spec = cal_spec(physician1_label, physician1_pred)
    physician2_sens = cal_sens(physician2_label, physician2_pred)
    physician2_spec = cal_spec(physician2_label, physician2_pred)
    physician3_sens = cal_sens(physician3_label, physician3_pred)
    physician3_spec = cal_spec(physician3_label, physician3_pred)
    physician4_sens = cal_sens(physician4_label, physician4_pred)
    physician4_spec = cal_spec(physician4_label, physician4_pred)
    physician5_sens = cal_sens(physician5_label, physician5_pred)
    physician5_spec = cal_spec(physician5_label, physician5_pred)

    physician_sen_mean = np.mean([physician1_sens, physician2_sens, physician3_sens, physician4_sens, physician5_sens])
    physician_spec_mean = np.mean([physician1_spec, physician2_spec, physician3_spec, physician4_spec, physician5_spec])
    physician_sen_CI1, physician_sen_CI2 = cal_CI([physician1_sens, physician2_sens, physician3_sens, physician4_sens, physician5_sens])
    physician_spec_CI1, physician_spec_CI2 = cal_CI([physician1_spec, physician2_spec, physician3_spec, physician4_spec, physician5_spec])

    physician_fpr_mean = 1 - physician_spec_mean
    physician_fpr_CI1, physician_fpr_CI2 = 1 - physician_spec_CI2, 1 - physician_spec_CI1

    all_csv_sens_mean.append(physician_sen_mean)
    all_csv_sens_CI.append([physician_sen_CI1, physician_sen_CI2])
    all_csv_fpr_mean.append(physician_fpr_mean)
    all_csv_fpr_CI.append([physician_fpr_CI1, physician_fpr_CI2])

fpr1, tpr1, thresholds1 = roc_curve(all_csv_label[0], all_csv_DLRN[0])
fpr2, tpr2, thresholds2 = roc_curve(all_csv_label[1], all_csv_DLRN[1])
fig, ax = plt.subplots(figsize=(7, 7))

ax.errorbar(all_csv_fpr_mean[0], all_csv_sens_mean[0],
            xerr=[[all_csv_fpr_mean[0] - all_csv_fpr_CI[0][0]], [all_csv_fpr_CI[0][1] - all_csv_fpr_mean[0]]],
            yerr=[[all_csv_sens_mean[0] - all_csv_sens_CI[0][0]], [all_csv_sens_CI[0][1] - all_csv_sens_mean[0]]],
            marker=',', color='#BC5259', elinewidth=2.5, capthick=1.5, capsize=4, label='Average reader in internal test cohort (95% CI)')
ax.plot(fpr1, tpr1,
        label='MMRN in internal test cohort\nAUC {:.3f} (95% CI: {:.3f}–{:.3f})'.format(all_csv_result[0][0], all_csv_result[0][1],
                                                                all_csv_result[0][2]),
        linewidth=2.5, color='#BC5259')

ax.errorbar(all_csv_fpr_mean[1], all_csv_sens_mean[1],
            xerr=[[all_csv_fpr_mean[1] - all_csv_fpr_CI[1][0]], [all_csv_fpr_CI[1][1] - all_csv_fpr_mean[1]]],
            yerr=[[all_csv_sens_mean[1] - all_csv_sens_CI[1][0]], [all_csv_sens_CI[1][1] - all_csv_sens_mean[1]]],
            marker=',', color='#5EA969', elinewidth=2.5, capthick=1.5, capsize=4, label='Average reader in external test cohort (95% CI)')
ax.plot(fpr2, tpr2,
        label='MMRN in external test cohort\nAUC {:.3f} (95% CI: {:.3f}–{:.3f})'.format(all_csv_result[1][0], all_csv_result[1][1],
                                                                all_csv_result[1][2]),
        linewidth=2.5, color='#5EA969')

# spe,sens
ax.errorbar(1-0.778, 0.774,
            marker='D', color='#BC5259', markersize=8, label='Youden\'s index bucket in internal test cohort')
ax.errorbar(1-0.778, 0.774,
            marker='D', color='#BC5259', markersize=12, alpha=0.2)

ax.errorbar(1-0.714, 0.780,
            marker='D', color='#5EA969', markersize=8, label='Youden\'s index bucket in external test cohort')
ax.errorbar(1-0.714, 0.780,
            marker='D', color='#5EA969', markersize=12, alpha=0.2)

# Aligned SPE
ax.errorbar(1-0.852, 0.578,
            marker='D', color='#8B0000', markersize=8, label='Aligned SPE bucket in internal test cohort')
ax.errorbar(1-0.852, 0.578,
            marker='D', color='#8B0000', markersize=12, alpha=0.2)

ax.errorbar(1-0.890, 0.576,
            marker='D', color='#006400', markersize=8, label='Aligned SPE bucket in external test cohort')
ax.errorbar(1-0.890, 0.576,
            marker='D', color='#006400', markersize=12, alpha=0.2)

# Aligned SEN
ax.errorbar(1-0.881, 0.513,
            marker='D', color='#FF9999', markersize=8, label='Aligned SEN bucket in internal test cohort')
ax.errorbar(1-0.881, 0.513,
            marker='D', color='#FF9999', markersize=12, alpha=0.2)

ax.errorbar(1-0.918, 0.507,
            marker='D', color='#99FF99', markersize=8, label='Aligned SEN bucket in external test cohort')
ax.errorbar(1-0.918, 0.507,
            marker='D', color='#99FF99', markersize=12, alpha=0.2)

x_start = 0  # 设置起始横坐标
x_end = 0.3    # 设置终止横坐标
y_start = 0.3  # 设置起始纵坐标
y_end = 1    # 设置终止纵坐标
plt.fill([x_start, x_end, x_end, x_start], [y_start, y_start, y_end, y_end], color='blue', alpha=0.1)
ax.set_xlabel('1-Specificity', fontsize=12)
ax.set_ylabel('Sensitivity', fontsize=12)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
# ax.set_title('ROC comparision')
ax.legend(fontsize=12)
plt.tight_layout()  # 调整布局
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.savefig(f'./img/ROC_compare_doctor/all_cohort_com_phy_test.png', dpi=1000)  # 保存为文件
plt.show()
