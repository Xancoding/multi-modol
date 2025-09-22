import csv
import config

def process_newborn_data(input_file, output_file=None, segment_duration=2.5):
    results = []
    total_segments_count = 0  # 总片段数量
    total_cry_duration = 0    # 总哭泣时长
    
    with open(input_file, 'r') as infile:
        for line in infile:
            row = line.strip().split('\t')
            if not row:  # 跳过空行
                continue
                
            sample_name = row[0]  # 第一列是文件名
            labels = row[1:]      # 后面的列才是标签
            
            # 检查标签是否有效（0或1）
            valid_labels = []
            for label in labels:
                if label.strip() in {'0', '1'}:  # 确保是 0 或 1
                    valid_labels.append(int(label))
                else:
                    print(f"警告：'{label}' 不是有效的标签（0或1），已忽略")
            
            # 查找所有标签为0的片段（0表示有哭声）
            segments = []
            current_start = None
            
            for i, label in enumerate(valid_labels):
                if label == 0:  # 当前是哭声
                    if current_start is None:  # 开始新的片段
                        current_start = i * segment_duration
                else:  # 当前不是哭声
                    if current_start is not None:  # 结束当前片段
                        segments.append((current_start, i * segment_duration))
                        current_start = None
            
            # 处理最后一个可能的片段
            if current_start is not None:
                segments.append((current_start, len(valid_labels) * segment_duration))
            
            # 统计当前样本的片段数量和时长
            sample_segment_count = len(segments)
            sample_cry_duration = sum([end - start for start, end in segments])
            
            total_segments_count += sample_segment_count
            total_cry_duration += sample_cry_duration
            
            # 格式化输出
            if segments:
                time_ranges = "、".join([f"{s[0]:.1f}-{s[1]:.1f}s" for s in segments])
                result = f"{sample_name} 有哭声的时间段: {time_ranges} (共{sample_segment_count}个片段，总时长{sample_cry_duration:.1f}秒)"
            else:
                result = f"{sample_name} 没有检测到哭声"
            
            results.append(result)
    
    # 打印统计信息
    print("="*80)
    print(f"统计信息:")
    print(f"总哭泣片段数量: {total_segments_count} 个")
    print(f"总哭泣时长: {total_cry_duration:.1f} 秒")
    print(f"平均每个片段时长: {total_cry_duration/max(total_segments_count, 1):.1f} 秒")
    print("="*80 + "\n")
    
    # 如果需要，写入输出文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write("婴儿哭声检测结果:\n")
            outfile.write("="*80 + "\n")
            for result in results:
                outfile.write(result + "\n")
           
process_newborn_data(config.data + '.csv', config.data + '.txt')