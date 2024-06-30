import os

import pandas as pd

img_name = os.listdir('../../../dataset/cropped/internal_test_cohort')

data = pd.read_csv('clinical_data_internal_test_cohort_mul.csv', usecols=[0])
data = data.iloc[:, 0].to_list()
data = [int(ids) for ids in data]
for name in img_name:
    id = int(name.replace('.jpg', ''))
    if id not in data:
        print(id)