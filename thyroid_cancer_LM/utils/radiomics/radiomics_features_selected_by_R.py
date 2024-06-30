#encoding=utf-8
import os

import pandas as pd
import numpy as np
# 根据R语言，LASSO回归得到的特征索引值来提取筛选后的特征
selected_features_coefs = ('4.286099e-03  '
                           '1.565463e-01 -8.575078e-01 '
                           '-1.815362e-16 -3.630723e-16  '
                           '2.645261e-03 -4.673805e-01 '
                           '1.363968e-01 -3.663373e-02 '
                           '-9.589274e-10 -1.699562e-05 '
                           '-6.225699e-04')
selected_features_indices = [205, 273, 374, 375, 376, 391, 501, 904, 912, 1019, 1021, 1031]
selected_features_indices = [indices-2 for indices in selected_features_indices]
selected_features_indices = sorted(list(set(selected_features_indices)))

radiomics_csv_name = ['radiomics_fat_primary_cohort.csv', 'radiomics_fat_internal_test_cohort.csv', 'radiomics_fat_external_test_cohort1.csv']
radiomics_csv_root_path = './radiomics_features'
radiomics_csv_output_path = './radiomics_features_selected'
for name in radiomics_csv_name:
    radiomic_csv_content_path = os.path.join(radiomics_csv_root_path, name)
    radiomics_csv_output_content_path = os.path.join(radiomics_csv_output_path, name)
    radiomic_csv = pd.read_csv(radiomic_csv_content_path)
    ids = radiomic_csv.iloc[:, 0]
    labels = radiomic_csv.iloc[:, 1]
    features = radiomic_csv.iloc[:, 2:]
    selected_features = features.iloc[:, selected_features_indices]
    final_data = {'ID': ids, 'Label': labels}
    df = pd.DataFrame(final_data)
    df = pd.concat([df, selected_features], axis=1)
    df.to_csv(radiomics_csv_output_content_path, encoding='GBK', index=False)

