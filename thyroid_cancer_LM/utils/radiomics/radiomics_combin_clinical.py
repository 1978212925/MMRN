import pandas as pd
import os
clinical_file_name_list = ['clinical_data_primary_cohort_mul.csv',
                           'clinical_data_internal_test_cohort_mul.csv',
                           'clinical_data_external_test_cohort1_mul.csv']
clinical_root_path = '../clinical_data/uni_mul_variate_clinical_data'

radiomics_file_name_list = ['radiomics_fat_primary_cohort.csv',
                            'radiomics_fat_internal_test_cohort.csv',
                            'radiomics_fat_external_test_cohort1.csv']

radiomics_root_path = '../radiomics/radiomics_features_selected'

for i in range(len(clinical_file_name_list)):
    clinical_data = pd.read_csv(os.path.join(clinical_root_path, clinical_file_name_list[i]))
    radiomics_data = pd.read_csv(os.path.join(radiomics_root_path, radiomics_file_name_list[i]), usecols=[0,2,3,4,5,6,7,8,9,10,11,12,13])

    merge_data = pd.merge(clinical_data, radiomics_data, how='left', on='ID')

    merge_data.to_csv(f'radiomics_combine_clinical/radiomics_combine_clinical_{i}.csv', index=False)