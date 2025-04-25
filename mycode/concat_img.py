import numpy as np
from PIL import Image
import os
from pathlib import Path


def ensure_directory_exists(directory):
    """
    检查文件夹是否存在，如果不存在则创建。
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_and_concatenate_images(output_path, *image_paths):
    """
    动态拼接任意数量的图片。
    :param output_path: 拼接后图片的保存路径
    :param image_paths: 可变参数，传入需要拼接的图片路径
    """
    ensure_directory_exists(os.path.dirname(output_path))

    # 读取所有图片并转换为 numpy 数组
    images_np = []
    for path in image_paths:
        if os.path.exists(path):  # 检查图片是否存在
            image = Image.open(path)
            images_np.append(np.array(image))
        else:
            print(f"警告：图片路径不存在，跳过 {path}")

    if not images_np:
        print("错误：没有有效的图片可以拼接。")
        return

    # 上下拼接图片
    concatenated_image_np = np.concatenate(images_np, axis=0)

    # 将 numpy 数组转换回 PIL 图像
    concatenated_image = Image.fromarray(concatenated_image_np)

    # 保存拼接后的图片
    concatenated_image.save(output_path)
    print(f"拼接后的图片已保存至: {output_path}")


if __name__ == "__main__":
    path = Path('output')
    babys = [item.name for item in path.iterdir() if item.is_dir()]

    # 只需在这里修改要拼接的图片列表
    image_paths = [
        'audio_energy.png',
        'video_optical_flow_std_origin.png',
        # 可以添加任意数量的图片文件名
        # 'audio_mfcc.png',
        # 'other_image.png',
    ]

    for baby in babys:
        # 直接使用所有图片路径进行拼接
        read_and_concatenate_images('concatenated.png', *image_paths)