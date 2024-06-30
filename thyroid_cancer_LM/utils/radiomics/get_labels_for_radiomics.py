import os
import pandas as pd
id_lables = pd.read_csv('../../dataset/thyroid_cancer_LM.csv', encoding='GBK')
radiomics_csv_name = ['radiomics_fat_primary_cohort.csv', 'radiomics_fat_internal_test_cohort.csv', 'radiomics_fat_external_test_cohort1.csv']
radiomics_csv_root_path = './radiomics_features'
for name in radiomics_csv_name:
    radiomics_csv_content_path = os.path.join(radiomics_csv_root_path, name)
    radiomics_csv = pd.read_csv(radiomics_csv_content_path)

    radiomics_csv_with_label = pd.merge(radiomics_csv, id_lables, on='ID', how='left')

    radiomics_csv_with_label.to_csv(radiomics_csv_content_path, encoding='GBK', index=False)