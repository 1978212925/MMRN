import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
aug_transform = transforms.Compose([
    # transforms.ColorJitter(
    #                         brightness=0.5,
    #                         contrast=0.5,
    #                         saturation=0.5,
    #                         hue=0.5),

])


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def plot_histogram(images, label):
    pixel_count = np.zeros(256)
    total_pixels = 0

    for img in images:
        img = img.convert('L')
        if label == "Primary cohort":
            img = aug_transform(img)
            img = F.adjust_brightness(img, 2)
        img = np.array(img)
        values, counts = np.unique(img.astype(int), return_counts=True)
        for value, count in zip(values, counts):
            if value >= 5 and value <= 250:
                pixel_count[value] += count
                total_pixels += count

    pixel_frequency = pixel_count / total_pixels
    plt.bar(range(256), pixel_frequency, label=label, alpha=0.7)


# 加载数据集
dataset1_path = '../dataset/merge_region/primary_cohort'
dataset2_path = '../dataset/merge_region/external_test_cohort1'

dataset1_images = load_images_from_folder(dataset1_path)
dataset2_images = load_images_from_folder(dataset2_path)

# 绘制直方图
plt.figure(figsize=(10, 5))
plot_histogram(dataset1_images, "Primary cohort")
plot_histogram(dataset2_images, "External test cohort")
plt.title('')
plt.xlabel('Pixel Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()
