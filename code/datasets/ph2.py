import os
import shutil

def extract_images():
    # 指定源文件夹路径和目标文件夹路径
    source_folder = 'E:\Dataset\PH2\PH2 Dataset images'  # 替换为你的源文件夹路径
    dermoscopic_folder = 'E:\Dataset\PH2\PH2\images'  # 替换为你的目标dermoscopic文件夹路径
    lesion_folder = 'E:\Dataset\PH2\PH2\masks'  # 替换为你的目标lesion文件夹路径

    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(dermoscopic_folder, exist_ok=True)
    os.makedirs(lesion_folder, exist_ok=True)

    # 遍历源文件夹中的所有子文件夹
    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)

        # 确保这是一个文件夹
        if os.path.isdir(subfolder_path):
            dermoscopic_path = os.path.join(subfolder_path, f"{subfolder}_Dermoscopic_Image")
            lesion_path = os.path.join(subfolder_path, f"{subfolder}_lesion")

            # 处理Dermoscopic_Image文件夹
            if os.path.isdir(dermoscopic_path):
                for file_name in os.listdir(dermoscopic_path):
                    if file_name.endswith('.bmp'):
                        src_file = os.path.join(dermoscopic_path, file_name)
                        dst_file = os.path.join(dermoscopic_folder, file_name)
                        shutil.copy(src_file, dst_file)
                        print(f"Copied {src_file} to {dst_file}")

            # 处理lesion文件夹
            if os.path.isdir(lesion_path):
                for file_name in os.listdir(lesion_path):
                    if file_name.endswith('.bmp'):
                        src_file = os.path.join(lesion_path, file_name)
                        dst_file = os.path.join(lesion_folder, file_name)
                        shutil.copy(src_file, dst_file)
                        print(f"Copied {src_file} to {dst_file}")

    print("图片提取完成。")

def divide_dataset():
    from PIL import Image
    import os
    import shutil
    import random

    # 设置文件夹路径
    images_folder = 'E:\Dataset\PH2\PH2_dataset\images'  # 替换为你的images文件夹路径
    masks_folder = 'E:\Dataset\PH2\PH2_dataset\masks'  # 替换为你的 masks 文件夹路径
    train_images_folder = r'E:\Dataset\PH2\PH2\train\images'  # 替换为你的训练 images 文件夹路径
    train_masks_folder =  r'E:\Dataset\PH2\PH2\train\masks'  # 替换为你的训练 masks 文件夹路径
    test_images_folder =  r'E:\Dataset\PH2\PH2\val\images'  # 替换为你的测试 images 文件夹路径
    test_masks_folder =   r'E:\Dataset\PH2\PH2\val\masks'  # 替换为你的测试 masks 文件夹路径

    # 确保目标文件夹存在
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_masks_folder, exist_ok=True)
    os.makedirs(test_images_folder, exist_ok=True)
    os.makedirs(test_masks_folder, exist_ok=True)

    # 获取图片文件名
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.bmp')]

    # 为每个图片生成对应的掩码文件名
    mask_files = [f.replace('.bmp', '_lesion.bmp') for f in image_files]

    # 确保每个掩码文件都存在
    for mask in mask_files:
        if mask not in os.listdir(masks_folder):
            raise ValueError(f"Mask file {mask} is missing in masks folder.")

    # 打乱文件顺序
    combined_files = list(zip(image_files, mask_files))
    random.shuffle(combined_files)

    # 计算分割点
    split_index = int(len(combined_files) * 0.7)

    # 分配文件到训练集和测试集
    train_files = combined_files[:split_index]
    test_files = combined_files[split_index:]

    # 转换图片并复制到训练集和测试集文件夹
    for image_file, mask_file in train_files:
        # 处理图片
        img_path = os.path.join(images_folder, image_file)
        img = Image.open(img_path).resize((256, 256))
        img.save(os.path.join(train_images_folder, image_file.replace('.bmp', '.jpg')), 'JPEG')

        # 处理掩码
        mask_path = os.path.join(masks_folder, mask_file)
        mask = Image.open(mask_path).resize((256, 256))
        mask.save(os.path.join(train_masks_folder, mask_file.replace('.bmp', '.jpg')), 'JPEG')

    for image_file, mask_file in test_files:
        # 处理图片
        img_path = os.path.join(images_folder, image_file)
        img = Image.open(img_path).resize((256, 256))
        img.save(os.path.join(test_images_folder, image_file.replace('.bmp', '.jpg')), 'JPEG')

        # 处理掩码
        mask_path = os.path.join(masks_folder, mask_file)
        mask = Image.open(mask_path).resize((256, 256))
        mask.save(os.path.join(test_masks_folder, mask_file.replace('.bmp', '.jpg')), 'JPEG')

    print("图片及掩码已按比例分配完成，并已转换为256x256大小的jpg格式。")

from datasets.dataset import NPY_datasets, Polyp_datasets, Isic_datasets
from torch.utils.data import DataLoader
from configs.isic16.config_setting_atrous import setting_config_atrousv2_ULPSR_step2_CNN_SE_SK
import numpy as np

def calculate_mean_std(train=True):
    """计算数据集的均值和标准差"""
    config = setting_config_atrousv2_ULPSR_step2_CNN_SE_SK

    dataset = Isic_datasets(r'E:\Dataset\PH2\PH2\\', config, train=train, test_dataset='isic16')
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    loader = DataLoader(dataset,
                              batch_size=20,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=4)

    # 用于存储所有图像的像素值
    all_pixels = []

    for images, _ in loader:
        images = images.float()  # 转换为浮点数
        all_pixels.append(images.view(-1).numpy())  # 将图像展平为一维数组并转换为 NumPy 数组

    all_pixels = np.concatenate(all_pixels)  # 将所有图像的像素值拼接起来
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)

    return mean, std



if __name__ == '__main__':
    divide_dataset()

    # mean, std = calculate_mean_std()
    # print(mean)
    # print(std)
    # mean, std = calculate_mean_std(train=False)
    # print(mean)
    # print(std)
