import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np

# 读取CSV文件
df = pd.read_csv('assets/for_tsfresh_data.csv')

# 选择要检查的id值
# ids_to_check = list(range(17, 22)) 
ids_to_check = [1,7,8,9]

# 设置图形大小
fig, axes = plt.subplots(len(ids_to_check), 1, figsize=(15, 10))

# 如果只有一个子图，将axes转换为列表，以便我们可以统一地处理它
if len(ids_to_check) == 1:
    axes = [axes]

for idx, current_id in enumerate(ids_to_check):
    # 根据当前id选择数据
    subset = df[df['id'] == current_id]
    
    # 使用Axes对象绘制子图
    ax = axes[idx]
    ax2 = ax.twinx()  # 创建第二个y轴
    
    for event_type, color, axis in [('npc_theta', 'b', ax2), ('brake', 'r', ax)]:
        signal = subset[event_type].values
        diff_signal = np.diff(signal)
        algo = rpt.Pelt(model="rbf").fit(diff_signal)
        result = algo.predict(pen=2)
        # print(f'result: {result}')
        axis.plot(signal, color+'-')  # 绘制信号
        
        for i in range(1, len(result)):
            start, end = result[i-1], result[i]
            change_rate = (signal[end-1] - signal[start]) / (end - start)
            change_magnitude = signal[end-1] - signal[start]#np.max(signal[start:end]) - np.min(signal[start:end])
            avg_value = np.mean(signal[start:end])
            
            # 标注在图上
            annotation_text = f"Rate: {change_rate:.2f}\nMagnitude: {change_magnitude:.2f}\nMean: {avg_value:.2f}"
            axis.annotate(annotation_text, xy=(start, signal[start]),
                          xytext=(0, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->", color=color),
                          fontsize=8, ha='center', color=color)
            axis.axvline(x=start, color=color, linestyle='--', alpha=0.5)  # 在result点画竖线
    
    ax.set_title(f"Breakpoints for id: {current_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Brake Value")
    ax2.set_ylabel("NPC Theta Value")

plt.tight_layout()
plt.show()
