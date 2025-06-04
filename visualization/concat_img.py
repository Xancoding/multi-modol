import numpy as np
from PIL import Image
import os
from pathlib import Path

def read_and_concatenate_images(output_path, *image_paths):
    """
    垂直拼接图片（自动调整宽度一致）
    :param output_path: 输出路径
    :param image_paths: 可变参数，传入图片路径列表
    """
    images_np = []
    base_width = None
    
    for path in image_paths:
        if os.path.exists(path):
            try:
                image = Image.open(path).convert('RGB')  # 统一为RGB模式
                if base_width is None:
                    base_width = image.width  # 设置基准宽度
                
                # 调整宽度（保持宽高比）
                if image.width != base_width:
                    ratio = base_width / float(image.width)
                    new_height = int(image.height * ratio)
                    image = image.resize((base_width, new_height), Image.LANCZOS)
                
                images_np.append(np.array(image))
            except Exception as e:
                print(f"错误：无法处理图片 {path} ({str(e)})")
        else:
            print(f"警告：图片路径不存在，跳过 {path}")
    
    if not images_np:
        print("错误：没有有效的图片可以拼接。")
        return
    
    # 垂直拼接
    concatenated_image_np = np.concatenate(images_np, axis=0)
    Image.fromarray(concatenated_image_np).save(output_path)
    print(f"图片已拼接保存至: {output_path}")

if __name__ == "__main__":
    image_paths = [
        'img/audio_spectrogram.jpg',
        # 'img/visualization_features.jpg'  # 确保文件存在
        '/root/mm/mycode/img/llj-baby-f_2025-03-19-13-09-15_cam1_visualization.jpg'
    ]
    read_and_concatenate_images('img/concatenated.jpg', *image_paths)