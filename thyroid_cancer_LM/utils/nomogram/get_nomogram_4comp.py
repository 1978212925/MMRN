# encoding=utf-8
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV



# 先训练每一个特征对应的Logist模型
for fold in range(1, 6):
    print(fold)
    nomogram_data = pd.read_csv(f'../nomogram/nomogram_signature/fold_{fold}/train_nom_sign_{fold}.csv')
    label = np.array(nomogram_data.iloc[:, 1]).astype(int)
    IPN_signature = np.reshape(np.array(nomogram_data.iloc[:, 2]).astype(float), (-1, 1))
    Gender_signature = np.reshape(np.array(nomogram_data.iloc[:, 3]).astype(float), (-1, 1))
    Age_signature = np.reshape(np.array(nomogram_data.iloc[:, 4]).astype(float), (-1, 1))
    T_stage_signature = np.reshape(np.array(nomogram_data.iloc[:, 5]).astype(float), (-1, 1))
    Size_of_US_signature = np.reshape(np.array(nomogram_data.iloc[:, 6]).astype(float), (-1, 1))
    Adipose_signature_1 = np.reshape(np.array(nomogram_data.iloc[:, 7]).astype(float), (-1, 1))
    Adipose_signature_2 = np.reshape(np.array(nomogram_data.iloc[:, 8]).astype(float), (-1, 1))
    Clinical_signature = np.concatenate(
        (Gender_signature, Age_signature, T_stage_signature, Size_of_US_signature), axis=1)
    Adipose_radiomic_signature = np.concatenate((Adipose_signature_1, Adipose_signature_2), axis=1)
    Clinical_adipose_signature = np.concatenate((Gender_signature, Age_signature, T_stage_signature,
                                                 Size_of_US_signature, Adipose_signature_1, Adipose_signature_2),
                                                axis=1)
    Clinical_IPN_signature = np.concatenate(
        (Gender_signature, Age_signature, T_stage_signature, Size_of_US_signature, IPN_signature), axis=1)
    Nomogram = np.concatenate(
        (Gender_signature, Age_signature, T_stage_signature, Size_of_US_signature, IPN_signature,
         Adipose_signature_1, Adipose_signature_2), axis=1)

    Clinical_Logist_regressor = LogisticRegression()
    Clinical_Logist_regressor.fit(Clinical_signature, label)
    joblib.dump(Clinical_Logist_regressor, f"./dif_features_combin_pkl/fold_{fold}/Clinical_Logist_regressor.pkl")

    IPN_Logist_regressor = LogisticRegression()
    IPN_Logist_regressor.fit(IPN_signature, label)
    joblib.dump(IPN_Logist_regressor, f"./dif_features_combin_pkl/fold_{fold}/IPN_Logist_regressor.pkl")

    Adipose_Logist_regressor = LogisticRegression()
    Adipose_Logist_regressor.fit(Adipose_radiomic_signature, label)
    joblib.dump(Adipose_Logist_regressor, f"./dif_features_combin_pkl/fold_{fold}/Adipose_Logist_regressor.pkl")

    Clinical_adipose_regressor = LogisticRegression()
    Clinical_adipose_regressor.fit(Clinical_adipose_signature, label)
    joblib.dump(Clinical_adipose_regressor, f"./dif_features_combin_pkl/fold_{fold}/Clinical_adipose_regressor.pkl")

    Clinical_IPN_regressor = LogisticRegression()
    Clinical_IPN_regressor.fit(Clinical_IPN_signature, label)
    joblib.dump(Clinical_IPN_regressor, f"./dif_features_combin_pkl/fold_{fold}/Clinical_IPN_regressor.pkl")

    Nomogram_regressor = LogisticRegression()
    Nomogram_regressor.fit(Nomogram, label)
    joblib.dump(Nomogram_regressor, f"./dif_features_combin_pkl/fold_{fold}/nomogram_regressor.pkl")

    all_csv_name = ['train', 'val', 'test0', 'test1']
    for csv_name in all_csv_name:
        print(csv_name)
        nomogram_data_2 = pd.read_csv(f'../nomogram/nomogram_signature/fold_{fold}/{csv_name}_nom_sign_{fold}.csv')
        img_id = np.array(nomogram_data_2.iloc[:, 0]).astype(int)
        label = np.array(nomogram_data_2.iloc[:, 1]).astype(int)
        IPN_signature = np.reshape(np.array(nomogram_data_2.iloc[:, 2]).astype(float), (-1, 1))
        Gender_signature = np.reshape(np.array(nomogram_data_2.iloc[:, 3]).astype(float), (-1, 1))
        Age_signature = np.reshape(np.array(nomogram_data_2.iloc[:, 4]).astype(float), (-1, 1))
        T_stage_signature = np.reshape(np.array(nomogram_data_2.iloc[:, 5]).astype(float), (-1, 1))
        Size_of_US_signature = np.reshape(np.array(nomogram_data_2.iloc[:, 6]).astype(float), (-1, 1))
        Adipose_signature_1 = np.reshape(np.array(nomogram_data_2.iloc[:, 7]).astype(float), (-1, 1))
        Adipose_signature_2 = np.reshape(np.array(nomogram_data_2.iloc[:, 8]).astype(float), (-1, 1))
        Clinical_signature = np.concatenate(
            (Gender_signature, Age_signature, T_stage_signature, Size_of_US_signature), axis=1)
        Adipose_radiomic_signature = np.concatenate((Adipose_signature_1, Adipose_signature_2), axis=1)
        Clinical_adipose_signature = np.concatenate((Gender_signature, Age_signature, T_stage_signature,
                                                     Size_of_US_signature, Adipose_signature_1, Adipose_signature_2),
                                                    axis=1)
        Clinical_IPN_signature = np.concatenate(
            (Gender_signature, Age_signature, T_stage_signature, Size_of_US_signature, IPN_signature), axis=1)
        Nomogram = np.concatenate(
            (Gender_signature, Age_signature, T_stage_signature, Size_of_US_signature, IPN_signature,
             Adipose_signature_1, Adipose_signature_2), axis=1)

        Clinical_Logist_regressor = joblib.load(f"./dif_features_combin_pkl/fold_{fold}/Clinical_Logist_regressor.pkl")

        IPN_Logist_regressor = joblib.load(f"./dif_features_combin_pkl/fold_{fold}/IPN_Logist_regressor.pkl")

        Adipose_Logist_regressor = joblib.load(f"./dif_features_combin_pkl/fold_{fold}/Adipose_Logist_regressor.pkl")

        Clinical_adipose_regressor = joblib.load(f"./dif_features_combin_pkl/fold_{fold}/Clinical_adipose_regressor.pkl")

        Clinical_IPN_regressor = joblib.load(f"./dif_features_combin_pkl/fold_{fold}/Clinical_IPN_regressor.pkl")

        Nomogram_regressor = joblib.load(f"./dif_features_combin_pkl/fold_{fold}/nomogram_regressor.pkl")

        Clinical_Logist_pred = Clinical_Logist_regressor.predict_proba(Clinical_signature)[:, 1]
        IPN_Logist_pred = IPN_Logist_regressor.predict_proba(IPN_signature)[:, 1]
        Adipose_Logist_pred = Adipose_Logist_regressor.predict_proba(Adipose_radiomic_signature)[:, 1]
        Clinical_adipose_pred = Clinical_adipose_regressor.predict_proba(Clinical_adipose_signature)[:, 1]
        Clinical_IPN_pred = Clinical_IPN_regressor.predict_proba(Clinical_IPN_signature)[:, 1]
        Nomogram_pred = Nomogram_regressor.predict_proba(Nomogram)[:, 1]

        print(roc_auc_score(label, Clinical_Logist_pred), roc_auc_score(label, IPN_Logist_pred), roc_auc_score(label, Adipose_Logist_pred)
              , roc_auc_score(label, Clinical_adipose_pred),roc_auc_score(label, Clinical_IPN_pred), roc_auc_score(label, Nomogram_pred))
        final_data = {'ID': img_id, 'Label': label, 'Clinical_pred':Clinical_Logist_pred, 'IPN_pred':IPN_signature.squeeze(), 'Adipose_pred':Adipose_Logist_pred,
                      'Clinical_adipose_pred':Clinical_adipose_pred, 'Clinical_IPN_pred':Clinical_IPN_pred, 'Nomogram_pred':Nomogram_pred}
        df = pd.DataFrame(final_data)
        df.to_csv(f'./dif_features_combin_result/fold_{fold}/{csv_name}_dif_features_com_result_{fold}.csv', index=False)
