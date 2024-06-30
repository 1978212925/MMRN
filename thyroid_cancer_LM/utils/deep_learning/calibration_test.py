import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
def log_transform(probs, alpha=1e-5):
    return np.log(probs + alpha)

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def temperature_scaling(logits, temperature):
    return np.exp(logits / temperature) / np.sum(np.exp(logits / temperature), axis=1, keepdims=True)

cohort_name = ['train', 'val', 'test0', 'test1']
for cohort in cohort_name:
    for i in range(1, 6):
        data = pd.read_csv(f'./node_fold_data/fold_{i}/{cohort}_DL_FC_result_{i}.csv')
        ids = data.iloc[:, 0]
        label = data.iloc[:, 1]
        y_pred = data.iloc[:, 2]

        fpr, tpr, thresholds = roc_curve(label, y_pred)
        print(roc_auc_score(label, y_pred))
        best_thresholds = thresholds[np.argmax(tpr - fpr)]
        y_pred_np = np.array(y_pred)
        label_np = np.array(label)
        y_pred_np = y_pred_np.reshape(-1, 1)

        Logist_regressor = LogisticRegression()
        # 假设 clf 是你的模型
        calibrated_clf = CalibratedClassifierCV(Logist_regressor, method='isotonic')
        calibrated_clf.fit(y_pred_np, label_np)
        y_pred2 = calibrated_clf.predict_proba(y_pred_np)[:, 1]
        y_pred2 = y_pred2.reshape(-1, 1)
        # 假设 prob 是模型的输出概率
        scaler = StandardScaler()
        normalized_probs = scaler.fit_transform(y_pred2)
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_probs = min_max_scaler.fit_transform(normalized_probs)
        y_pred3 = scaled_probs
        fpr2, tpr2, thresholds2 = roc_curve(label, y_pred3)
        best_thresholds2 = thresholds2[np.argmax(tpr2 - fpr2)]
        print(roc_auc_score(label, y_pred3.reshape(-1)))
        final_data = {'ID':ids, 'Label':label,'Y_pred':y_pred3.reshape(-1)}
        pd_data = pd.DataFrame(final_data)

        pd_data.to_csv(f'./node_fold_data/fold_{i}/{cohort}_DL_FC_result_{i}.csv', index=False)