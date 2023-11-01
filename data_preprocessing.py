import numpy as np
import csv
import pandas as pd
import re
import os



def npc_from_txt_wid(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    blocks = re.split(r'module_name: "perception_obstacle"', content)

    # 创建一个空的DataFrame用于存储数据
    df = pd.DataFrame()
    
    for block in blocks:
        data = dict()
        ts_match = re.search(r'timestamp_sec: ([\d.+-]+)', block)
        if ts_match:
            ts = float(ts_match.group(1))   
            data['npc_ts'] = ts
        subblocks = re.split(r'perception_obstacle \{', block)
        # 计算并打印perception_obstacle的数量
        num_perception_obstacles = len(subblocks) - 1
        for i, subblock in enumerate(subblocks):
            # 使用正则表达式提取所需的数据
            id_match = re.search(r'id: (\d+)', subblock)
            x_match = re.search(r'position \{\s+x: ([\d.+-]+)', subblock)
            y_match = re.search(r'y: ([\d.+-]+)', subblock)
            theta_match = re.search(r'theta: ([\d.+-]+)', subblock)
            
            # 如果在块中找到了所有所需的数据，则将其添加到DataFrame中
            if id_match and x_match and y_match and theta_match:
                npc_id = int(id_match.group(1))
                x = float(x_match.group(1))
                y = float(y_match.group(1))
                theta = float(theta_match.group(1))

                # 动态创建列名
                data.update({
                    f'NPC{npc_id}X': x,
                    f'NPC{npc_id}Y': y,
                    f'NPC{npc_id}Theta': theta,
                })

        if data:
            # 使用pandas.concat将数据添加到DataFrame中
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    
    # 使用timestamp_sec列对DataFrame进行排序（如果需要的话）
    # df = df.sort_values(by='timestamp_sec').reset_index(drop=True)
    
    return df

def ego_from_txt(filepath):
    f = open(filepath, 'r')
    fframe = f.readlines()
    
    #append timestamp
    df_ts = pd.DataFrame(columns=['ego_ts'])
    #append position
    df_lc = pd.DataFrame(columns=['egoX','egoY','heading'])
    
    for row in fframe:
        if row.find('timestamp_sec') != -1:
            ts = re.findall("(?<=[AZaz])?[0-9.+-]+",row) #exclude "timestamp_sec:"
            ts = float(ts[0]) #convert to data format
            l_no = fframe.index(row)
            
            lc_x = re.findall("(?<=[AZaz])?[0-9.+-]+",fframe[l_no + 6])
            lc_x = float(lc_x[0])
            lc_y = re.findall("(?<=[AZaz])?[0-9.+-]+",fframe[l_no + 7])
            lc_y = float(lc_y[0])
            heading = re.findall("(?<=[AZaz])?[0-9.+-]+",fframe[l_no +31])
            heading = float(heading[0])
            
            d_0 = {'ego_ts':[ts]}
            d_df0 = pd.DataFrame(data=d_0)
            df_ts = pd.concat([df_ts, d_df0])

            d_1 = {'egoX':[lc_x], 'egoY':[lc_y],'heading':[heading]}
            d_df1 = pd.DataFrame(data=d_1)
            df_lc = pd.concat([df_lc, d_df1])
    df = pd.concat([df_ts, df_lc], axis=1)

    return df

def cmd_from_txt(filepath):
    f = open(filepath, 'r')
    fframe = f.readlines()
    
    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=['cmd_ts', 'throttle', 'brake', 'steering'])
    
    for l_no, row in enumerate(fframe):
        if re.match(r'^header {', row):
            ts = re.findall("(?<=[AZaz])?[0-9.+-]+", fframe[l_no + 1])
            ts = float(ts[0])

            throttle = re.findall("(?<=[AZaz])?[0-9.+-]+", fframe[l_no + 11])
            if throttle:
                throttle = float(throttle[0])
            else:
                throttle = re.findall("(?<=[AZaz])?[0-9.+-]+", fframe[l_no + 12])
                if throttle:
                    throttle = float(throttle[0])
                else:
                    print(f"Error in {filepath}, line {l_no}: Throttle value not found")
                    print(f"timestamp = {ts}")

            # throttle = float(throttle[0])

            brake = re.findall("(?<=[AZaz])?[0-9.+-]+", fframe[l_no + 12])
            brake = float(brake[0])

            steering = re.findall("(?<=[AZaz])?[0-9.+-]+", fframe[l_no + 14])
            steering = float(steering[0])

            col_df = pd.concat([pd.DataFrame({'cmd_ts': [ts]}), 
                                 pd.DataFrame({'throttle': [throttle]}), 
                                 pd.DataFrame({'brake': [brake]}), 
                                 pd.DataFrame({'steering': [steering]})], axis=1)
            
            df = pd.concat([df, col_df], ignore_index=True)

    return df

def align_and_average(serial):
    data_path = root_path
    df_npc = npc_from_txt_wid(data_path+f'/txt_record/{serial}_obstacle.txt')
    df_ego = ego_from_txt(data_path+f'/txt_record/{serial}_pose.txt')
    df_cmd = cmd_from_txt(data_path+f'/txt_record/{serial}_control.txt')
## 根据是否是obstacle来的选择要不要手动对齐数据
    delta_ts = df_ego['ego_ts'].iloc[0] - df_npc['npc_ts'].iloc[0]
    df_npc['npc_ts'] += delta_ts

    # 使用列表来收集有效的数据
    aligned_ego_list = []
    avg_cmd_list = []
    aligned_npc_list = []
    
    # 上一个npc时间戳的索引
    last_npc_idx = 0
    
    for i in range(df_npc.shape[0]):
        # 对齐ego的时间戳
        ego_indices = np.where(df_ego['ego_ts'] < df_npc['npc_ts'].iloc[i])[0]
        if ego_indices.size == 0:
            continue
        ego_idx = ego_indices[-1]
        
        # 平均cmd的数据
        cmd_indices = np.where((df_cmd['cmd_ts'] >= df_npc['npc_ts'].iloc[last_npc_idx]) & 
                               (df_cmd['cmd_ts'] < df_npc['npc_ts'].iloc[i]))[0]
        
        # 检查cmd_indices是否为空
        if cmd_indices.size > 0:
            aligned_ego_list.append(df_ego.iloc[ego_idx])
            avg_cmd_list.append(df_cmd.iloc[cmd_indices].mean())
            aligned_npc_list.append(df_npc.iloc[i])
        
        last_npc_idx = i

    # 将列表转换为DataFrame
    df_aligned_ego = pd.DataFrame(aligned_ego_list).reset_index(drop=True)
    df_avg_cmd = pd.DataFrame(avg_cmd_list).reset_index(drop=True)
    df_aligned_npc = pd.DataFrame(aligned_npc_list).reset_index(drop=True)

    return df_aligned_ego, df_aligned_npc, df_avg_cmd


def process_serials(serials):
    merge_df = pd.DataFrame()
    
    for i in range(serials):
        # if i == 46:
        #     continue
        print(f"Processing serial {i}")
        aligned_ego, aligned_npc, avg_cmd = align_and_average(i)
        # 创建一个临时的DataFrame来存储这一轮的数据
        tmp = aligned_ego['ego_ts'].to_frame(name='time')
        tmp['id'] = i + 1  # 添加新列
        tmp = pd.concat([tmp, aligned_ego, aligned_npc, avg_cmd], axis=1)
        
        # 在尝试删除列之前，检查它们是否存在
        columns_to_drop = ['ego_ts', 'cmd_ts', 'npc_ts']
        for col in columns_to_drop:
            if col not in tmp.columns:
                print(f"Warning: Column '{col}' not found in DataFrame at iteration {i}. Available columns: {tmp.columns}")
            else:
                tmp = tmp.drop(columns=col)
        # 将这一轮的数据添加到merge_df中
        merge_df = pd.concat([merge_df, tmp], ignore_index=True)
    return merge_df


# 定义一个函数来解析文件并返回一个DataFrame
def parse_simulation_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 初始化空的列表来存储数据
    data = []
    
    # 初始化一个空的字典来临时存储每次迭代的数据
    iteration_data = {}
    
    # 遍历文件的每一行
    for line in lines:
        # 使用正则表达式来解析数据
        if "Iteration" in line:
            # 如果字典不为空，则将其添加到数据列表中，并清空字典
            if iteration_data:
                data.append(iteration_data)
                iteration_data = {}
            iteration_data['id'] = int(re.search(r'\d+', line).group())
        elif "Weather" in line:
            weather_data = re.findall(r'(\w+)=([\d.]+)', line)
            for item in weather_data:
                iteration_data[item[0]] = float(item[1])
        elif "Time of day" in line:
            iteration_data['daytime'] = float(re.search(r'[\d.]+', line).group())
        elif "Collision" in line:
            match = re.search(r"Collision: (True|False)", line)
            if match:
                # 如果找到了匹配的字符串，获取True或False的值
                value = match.group(1)
                # 根据True或False的值进行转换，并保存在字典中
                iteration_data['result'] = 1 if value == "True" else 0
    
    # 确保最后一次迭代的数据也被添加到数据列表中
    if iteration_data:
        data.append(iteration_data)
    
    # 将数据列表转换为DataFrame
    df = pd.DataFrame(data)
    
    return df



### 处理文件路径
root_path = 'assets/1failtoyield/'
# 提取结果等静态部分 使用函数解析文件并获取DataFrame
file_path = root_path+'simulation_data.txt'
df = parse_simulation_data(file_path)

# 显示DataFrame的前几行
# print(df)
df.to_csv(root_path+'features.csv')