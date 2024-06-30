import os
import pandas as pd
cohort_name = ['external_test_cohort1']
root_path = '../../dataset/cropped'
all_csv = pd.read_csv('./1y.csv', encoding='GBK')
for name in cohort_name:
    cohort_content_path = os.path.join(root_path, name)
    print(name)
    img_name_list = os.listdir(cohort_content_path)
    print(len(img_name_list))
    img_name_list = [int(name.replace('.jpg', '')) for name in img_name_list]
    splitted_csv = all_csv[all_csv['ID'].isin(img_name_list)]

    splitted_csv.to_csv(f'./splitted_clinical_data/clinical_data_{name}.csv', encoding='GBK', index=False)