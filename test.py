import json
import os
from glob import glob

def compare_face_features(file1, file2):
    """
    比较两个JSON文件中的Face特征是否一致
    
    参数:
        file1 (str): 第一个JSON文件路径
        file2 (str): 第二个JSON文件路径
        
    返回:
        bool: 如果所有Face特征一致返回True，否则返回False
        str: 不一致的详细信息
    """
    try:
        # 读取并解析两个JSON文件
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
        
        # 检查两个文件是否都有features字段
        if "features" not in data1 or "features" not in data2:
            return False, "缺少features字段"
        
        features1 = data1["features"]
        features2 = data2["features"]
        
        # 检查features长度是否相同
        if len(features1) != len(features2):
            return False, f"features长度不同: {len(features1)} vs {len(features2)}"
        
        # 遍历每个frame比较Face内容
        for i in range(len(features1)):
            frame1 = features1[i]
            frame2 = features2[i]
            
            # 检查Frame编号是否一致
            if frame1.get("Frame") != frame2.get("Frame"):
                return False, f"第{i}个frame编号不一致: {frame1.get('Frame')} vs {frame2.get('Frame')}"
            
            # 比较Face内容
            face1 = frame1.get("Face", [])
            face2 = frame2.get("Face", [])
            if face1 != face2:
                return False, f"Frame {frame1.get('Frame')}的Face内容不一致:\n{face1}\nvs\n{face2}"

            # 比较Left-arm内容
            left_arm1 = frame1.get("Left-arm", [])
            left_arm2 = frame2.get("Left-arm", [])
            if left_arm1 != left_arm2:
                return False, f"Frame {frame1.get('Frame')}的Left-arm内容不一致:\n{left_arm1}\nvs\n{left_arm2}"
            
            # 比较Right-arm内容
            right_arm1 = frame1.get("Right-arm", [])
            right_arm2 = frame2.get("Right-arm", [])
            if right_arm1 != right_arm2:
                return False, f"Frame {frame1.get('Frame')}的Right-arm内容不一致:\n{right_arm1}\nvs\n{right_arm2}"
            
            # 比较Left-leg内容
            left_leg1 = frame1.get("Left-leg", [])
            left_leg2 = frame2.get("Left-leg", [])
            if left_leg1 != left_leg2:
                return False, f"Frame {frame1.get('Frame')}的Left-leg内容不一致:\n{left_leg1}\nvs\n{left_leg2}"
            
            # 比较Right-leg内容
            right_leg1 = frame1.get("Right-leg", [])
            right_leg2 = frame2.get("Right-leg", [])
            if right_leg1 != right_leg2:
                return False, f"Frame {frame1.get('Frame')}的Right-leg内容不一致:\n{right_leg1}\nvs\n{right_leg2}"
        
        return True, "所有内容一致"
    
    except Exception as e:
        return False, f"比较过程中发生错误: {str(e)}"

# 使用示例
if __name__ == "__main__":
    dir = "/data/Leo/mm/data/Newborn200"
    # 获取所有JSON文件
    files = glob(os.path.join(dir, "Body", "*motion_features.json"), recursive=True)

    true_num = 0
    false_num = 0
    for file1 in files:
        file2 = file1.replace("Body", "NewBody")
 
        result, message = compare_face_features(file1, file2)
        # print("\n比较结果:")
        # print(message)
        if result:
            true_num += 1
        else:
            false_num += 1
    print(f"一致的文件数量: {true_num}")
    print(f"不一致的文件数量: {false_num}")
