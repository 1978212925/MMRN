import pandas as pd

clinical_fat_data = pd.read_csv('all_radiomics_combine_clinical_0.csv', encoding='GBK')
all_csv_name = ['train', 'val', 'test0', 'test1']

for fold in range(1, 6):
    for csv_name in all_csv_name:
        data = pd.read_csv(f'../nomogram_signature/fold_{fold}/{csv_name}_DL_FC_result_{fold}.csv')
        # 合并 data 和 age_sex_data 基于共同的 id 列
        merged_data = pd.merge(data, clinical_fat_data, on='ID', how='left')

        # 将合并后的数据存储到新的文件
        merged_data.to_csv(f'../nomogram_signature/fold_{fold}/{csv_name}_nom_sign_{fold}.csv', index=False)
