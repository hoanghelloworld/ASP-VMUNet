from PIL import Image
import os

# 输入和输出文件夹路径
input_folder = r'E:\Dataset\ISIC2016\ISBI2016_ISIC_Part1_Test_Data'
output_folder = r'E:\Dataset\ISIC2016\isic2016\val\images'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中所有的JPEG图片文件名
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
# image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# 循环处理每张图片
for image_file in image_files:
    try:
        # 打开图片文件
        img_path = os.path.join(input_folder, image_file)
        img = Image.open(img_path)

        # 调整图片大小为256x256像素
        img = img.resize((256, 256))

        # 构造输出文件路径
        output_path = os.path.join(output_folder, image_file)

        # 保存调整大小后的图片
        img.save(output_path)

        print(f"Processed {image_file}")

    except Exception as e:
        print(f"Failed to process {image_file}: {e}")
