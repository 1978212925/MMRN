import os
import cv2
import radiomics.featureextractor
import pandas as pd
import numpy as np
import SimpleITK as sitk


img_root_path = "../../image_preprocessing_toolkit/fat/cropped_img"
img_name = os.listdir(img_root_path)
mask_root_path = "../../image_preprocessing_toolkit/fat/cropped_mask"
feature_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
feature_extractor.enableAllFeatures()
feature_extractor.enableAllImageTypes()
# feature_extractor.enableImageTypeByName('Original')




# 创建一个空的DataFrame来存储特征
feature_df = pd.DataFrame()

for name in img_name:
    img_path = os.path.join(img_root_path, name)
    mask_path = os.path.join(mask_root_path, name)
    try:
        print(name)
        img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        mask[mask < 200] = 0
        mask[mask >= 200] = 1
        # 将numpy数组转换为SimpleITK图像对象
        img_sitk = sitk.GetImageFromArray(img)
        mask_sitk = sitk.GetImageFromArray(mask)
        radiomic_features = feature_extractor.execute(img_sitk, mask_sitk)
        radiomic_features['ID'] = name.replace(".png", "")
        radiomic_features.move_to_end('ID', last=False)
        # 将特征转换为DataFrame的行
        feature_row = pd.DataFrame([radiomic_features])
        # 将特征行添加到DataFrame中
        feature_df = feature_df._append(feature_row, ignore_index=True)
    except Exception as e:
        print(f"Error processing image {name}: {e}")

# 存储为CSV文件
feature_df['ID'] = feature_df['ID'].astype(str)
feature_df.to_csv('./radiomics_features/radiomics_fat.csv', index=False)
