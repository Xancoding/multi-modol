import os
import shutil

def move_files(source_folder, target_folder, files_to_move):
    """
    将指定文件从源文件夹移动到目标文件夹
    
    参数:
        source_folder (str): 源文件夹路径
        target_folder (str): 目标文件夹路径
        files_to_move (list): 要移动的文件名列表
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for filename in files_to_move:
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
            print(f"已移动: {filename}")
        else:
            print(f"文件不存在: {filename}")

# 示例使用
if __name__ == "__main__":

    infants = [
'hbxd-m_2025-07-29-14-45-40',
'hbxd-m_2025-07-29-15-06-59',
'thz-m_2025-08-05-12-45-55',
'hyy-baby-m_2025-08-05-13-29-37',
'dxh-baby-f_2025-08-05-13-48-59',
'cjyd-m_2025-08-05-15-43-31',
'ysqd-f_2025-07-29-16-34-26',
'ydw-baby-f_2025-07-29-15-17-50',
'mxt-baby-f_2025-07-29-13-09-45',
'whq-baby-m_2025-07-29-11-48-29',
'lj-baby-m_2025-07-29-12-06-14',
'hmx-baby-f_2025-08-14-16-54-55',   
'zm-baby-m_2025-07-23-19-24-10',
'wxl-baby-m_2025-08-05-12-19-40',
'xgx-baby-m_2025-07-29-15-39-53',
'hym-baby-f_2025-07-29-15-26-27',
'lxt-f_2025-07-29-17-14-18',
'ysqd-f_2025-07-29-15-32-40',
'cfy-baby-m_2025-08-05-14-05-54',     
    ]


    # dir = ['Body', 'Face', 'data']
    dir = 'data'
    avi_files = [infant + '.avi' for infant in infants]
    wav_files = [infant + '.wav' for infant in infants]
    files = avi_files + wav_files    

    # dir = 'Face'
    # json_files = [infant + '_face_landmarks.json' for infant in infants]
    # avi_files = [infant + '_face_landmarks.avi' for infant in infants]
    # files = json_files + avi_files

    # dir = 'Body'
    # json_files = [infant + '_motion_features.json' for infant in infants]
    # avi_files = [infant + '_masked.avi' for infant in infants]
    # files = json_files + avi_files

    source = os.path.join("/data/Leo/mm/data/NICU50", dir)
    target = os.path.join("/data/Leo/mm/data/badNICU50", dir)

    move_files(source, target, files)