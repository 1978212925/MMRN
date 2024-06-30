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


fig, ax = plt.subplots(figsize=(7, 7))

all_csv_name = ['test0', 'test1']

all_csv_result = [[0.829, 0.822, 0.835],
                  [0.818, 0.808, 0.828]]
color = ['#BC5259', '#5EA969']
name = ['internal test cohort', 'external test cohort']
CI = [[1 - 0.778, 0.774],
      [1 - 0.714, 0.780]]
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
    fpr1, tpr1, thresholds1 = roc_curve(csv_label, csv_DLRN)

    # 医师1
    # 绘制中空圆形
    ax.plot(1 - physician1_spec, physician1_sens, marker='o', markerfacecolor='none', markeredgecolor=color[i],
            markersize=18,
            markeredgewidth=3,
            label=f'Individual readers in {name[i]}')
    # 在中心添加文本（数字）
    ax.text(1 - physician1_spec, physician1_sens - 0.001, '1', ha='center', va='center', color=color[i], fontsize=15)

    # 医师2
    # 绘制中空圆形
    ax.plot(1 - physician2_spec, physician2_sens, marker='o', markerfacecolor='none', markeredgecolor=color[i],
            markersize=18,
            markeredgewidth=3, )
    # 在中心添加文本（数字）
    ax.text(1 - physician2_spec, physician2_sens - 0.001, '2', ha='center', va='center', color=color[i], fontsize=15)

    # 医师3
    # 绘制中空圆形
    ax.plot(1 - physician3_spec, physician3_sens, marker='o', markerfacecolor='none', markeredgecolor=color[i],
            markersize=18,
            markeredgewidth=3, )
    # 在中心添加文本（数字）
    ax.text(1 - physician3_spec, physician3_sens - 0.001, '3', ha='center', va='center', color=color[i], fontsize=15)

    # 医师4
    # 绘制中空圆形
    ax.plot(1 - physician4_spec, physician4_sens, marker='o', markerfacecolor='none', markeredgecolor=color[i],
            markersize=18,
            markeredgewidth=3, )
    # 在中心添加文本（数字）
    ax.text(1 - physician4_spec, physician4_sens - 0.001, '4', ha='center', va='center', color=color[i], fontsize=15)

    # 医师5
    # 绘制中空圆形
    ax.plot(1 - physician5_spec, physician5_sens, marker='o', markerfacecolor='none', markeredgecolor=color[i],
            markersize=18,
            markeredgewidth=3, )
    # 在中心添加文本（数字）
    ax.text(1 - physician5_spec, physician5_sens - 0.001, '5', ha='center', va='center', color=color[i], fontsize=15)

    ax.plot(fpr1, tpr1,
            label='MMRN in {:s}'.format(name[i], all_csv_result[0][0],
                                        all_csv_result[0][1],
                                        all_csv_result[0][2]),
            linewidth=2.5, color=color[i])

    ax.errorbar(CI[i][0], CI[i][1],
                marker='D', color=color[i], markersize=11, label=f'Youden\'s index bucket in {name[i]}')
    ax.errorbar(CI[i][0], CI[i][1],
                marker='D', color=color[i], markersize=15, alpha=0.2)

ax.set_xlabel('1-Specificity', fontsize=12)
ax.set_ylabel('Sensitivity', fontsize=12)

# Aligned SPE
ax.errorbar(1 - 0.855, 0.578,
            marker='D', color='#8B0000', markersize=11, label='Aligned SPE bucket in internal test cohort')
ax.errorbar(1 - 0.855, 0.578,
            marker='D', color='#8B0000', markersize=15, alpha=0.2)

ax.errorbar(1 - 0.890, 0.576,
            marker='D', color='#006400', markersize=11, label='Aligned SPE bucket in external test cohort')
ax.errorbar(1 - 0.890, 0.576,
            marker='D', color='#006400', markersize=15, alpha=0.2)

# Aligned SEN
ax.errorbar(1 - 0.885, 0.513,
            marker='D', color='#FF9999', markersize=11, label='Aligned SEN bucket in internal test cohort')
ax.errorbar(1 - 0.885, 0.513,
            marker='D', color='#FF9999', markersize=15, alpha=0.2)

ax.errorbar(1 - 0.918, 0.507,
            marker='D', color='#99FF99', markersize=11, label='Aligned SEN bucket in external test cohort')
ax.errorbar(1 - 0.918, 0.507,
            marker='D', color='#99FF99', markersize=15, alpha=0.2)


ax.set_xlim([0, 0.3])
ax.set_ylim([0.3, 1])
# ax.set_title('ROC comparision')
ax.legend(fontsize=12)
plt.tight_layout()  # 调整布局

plt.savefig(f'./img/ROC_compare_doctor/enlarger_all_cohort_com_phy_test.png', dpi=1000)  # 保存为文件
plt.show()
