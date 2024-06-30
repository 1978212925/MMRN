import os
import pandas as pd
cohort_name = ['internal_test_cohort']
root_path = '../../dataset/cropped'
all_csv = pd.read_csv('./radiomics_features/radiomics_fat.csv', encoding='GBK')
for name in cohort_name:
    cohort_content_path = os.path.join(root_path, name)
    img_name_list = os.listdir(cohort_content_path)
    img_name_list = [int(name.replace('.jpg', '')) for name in img_name_list]
    splitted_csv = all_csv[all_csv['ID'].isin(img_name_list)]

    splitted_csv.to_csv(f'./radiomics_features/radiomics_fat_{name}.csv', encoding='GBK', index=False)