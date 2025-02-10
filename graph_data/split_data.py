import random

# 读取原文件
def split_file(input_file, output_file_1, output_file_2, ratio=0.09):
    # 读取文件中的所有行
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 计算拆分的行数
    total_lines = len(lines)
    split_index = int(total_lines * ratio)
    
    # 打乱行顺序（可选，但为了随机拆分更公平）
    random.shuffle(lines)
    
    # 拆分成两个部分
    split_1 = lines[:split_index]
    split_2 = lines[split_index:]
    
    # 写入拆分后的文件
    with open(output_file_1, 'w') as f1:
        f1.writelines(split_1)
    
    with open(output_file_2, 'w') as f2:
        f2.writelines(split_2)

# 使用示例
input_file = 'graph_data/douban/dictionaries/groundtruth'
output_file_1 = 'graph_data/douban/dictionaries/node,split=0.09.train.dict'
output_file_2 = 'graph_data/douban/dictionaries/node,split=0.09.test.dict'
split_file(input_file, output_file_1, output_file_2)
