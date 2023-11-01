import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import os

### 需要处理的
iters_to_process = [1]
root_path = 'assets/1npccutin/'

# 创建一个有向图
G = nx.DiGraph()
event_names=set()
event_starts=dict()
for iter in iters_to_process:
    # 读取存储的CSV文件
    attributes_df = pd.read_csv(root_path+str(iter)+'as_attributes_fi.csv')
    event_df = pd.read_csv(root_path+str(iter)+'average_sequence.csv')
    # print(attributes_df,event_df)
    # 提取所有的事件名
    tmp = set([re.match(r"([a-zA-Z\d]+_\d+)", col).group(1) for col in attributes_df.columns if re.match(r"([a-zA-Z\d]+_\d+)", col)])
    event_names.update(tmp)
    event_start = {event: attributes_df[event + '_start'].iloc[0] for event in tmp}
    event_starts.update(event_start)




event_starts["initial"] = 0
event_names.add("initial")
# 添加"collision"事件
collision_start = len(event_df)-1
event_starts["collision"] = collision_start
event_chain = sorted(event_names, key=lambda x: event_starts[x])
event_chain.append("collision")  # 将"collision"事件添加到事件链的末尾

# 添加边
for i in range(len(event_chain)-1):
    G.add_edge(event_chain[i], event_chain[i+1])


# 设置图形的大小

plt.figure(figsize=(15, 5))
# 定义节点位置
pos = {event: (start/2, 0) for event, start in event_starts.items()}

# # 获取每个事件的重要性
# importances = [attributes_df[f"{event}_importance"].iloc[0] for event in event_chain[:-1]]

# # 使用colormap来映射重要性到颜色
# cmap = plt.cm.Blues  # 使用蓝色系列
node_colors = ['lightblue']*(len(event_chain)-1)
node_colors.append('pink')  # 或'lightblue'，取决于你想要的颜色


labels = {}
for node in G.nodes():
    # 如果节点名称以 "_数字" 结尾，则截取名称的前部分，否则保持名称不变
    match = re.match(r"(.+?)_\d+$", node)
    if match:
        labels[node] = match.group(1)
    else:
        labels[node] = node
# 绘制图形
nx.draw(G, pos, labels=labels, node_color=node_colors, node_size=6000, edge_color='gray', linewidths=2, font_size=25, arrowsize=30, connectionstyle='arc3,rad=0')


for event in event_chain:
    if event == "collision":
        # 为"collision"事件添加一个简单的标签
        label = f"Time: {collision_start}\n"
    else:
        attributes = attributes_df.iloc[0]  # 选择第一行的属性作为示例
        label = ""  # 初始化一个空标签
        for attr, value in attributes.items():
            if event in attr:  # 检查属性是否属于当前事件
                # 修改属性名
                attr_name = attr.split('_')[-1]
                if attr_name == 'start':
                    attr_name = 'Time'
                else:
                    attr_name = attr_name.capitalize()  # 将首字母更改为大写
                
                # 添加属性信息到标签
                label += f"{attr_name}: {value:.2f}\n"

    # 在指定位置添加文本
    plt.text(pos[event][0], pos[event][1], label, fontsize=20, ha='right')


# 轴上轴TODO:iters_to_process[0]
item = iters_to_process[0] ##
vertical_data_df = pd.read_csv(root_path+str(item)+'average_sequence.csv')
feature_series = vertical_data_df.iloc[:, 0].values  
offset = -min(feature_series) + 0.1  # 偏移量为数据的最小值的负数加上一个小的常数
max_value = max(feature_series)  # 获取最大值

# 创建 x 和 y 坐标的列表
x_values = [i/2 for i in range(len(feature_series))]
y_values = [(value + offset)/2 for value in feature_series]  # 使用计算出的偏移量

# 绘制连线
plt.plot(x_values, y_values, color='pink')
plt.arrow(0, 0, 0, (max_value + offset)/1 - 0.2, head_width=0.4, head_length=0.05, fc='black', ec='black') # 在x=0的位置画一条黑色的竖线作为纵轴
plt.ylim(-0.1, max_value + offset)  # some_value是你想要更多显示的底部区域的值
plt.text(-0.5, max_value + offset - 0.2, vertical_data_df.columns[0], fontsize=14, rotation=0, ha='right')  # 你可以根据需要调整文本位置和其他属性


file_path = os.path.join(root_path+'mini_plots/', f"event_chain.png")  # 创建完整的文件路径
plt.savefig(file_path)  # 保存图形

# 显示图形
plt.show()
