import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import cdist_dtw
from scipy.cluster.hierarchy import  linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import ruptures as rpt
import os
import re
import random

def hierarchical_clustering_and_reference(one_label_all_series, threshold=5):
    """
    Perform hierarchical clustering on a list of time series and select a reference series.
    
    Parameters:
    - one_label_all_series: list of np.array
        The time series to be clustered.
    - threshold: float, optional (default=10)
        The threshold to use for forming flat clusters.
    
    Returns:
    - labels: np.array
        The cluster labels for each time series.
    - reference_sequence: np.array
        A reference time series selected from the most common cluster.
    """

    for idx, ts in enumerate(one_label_all_series):
        if len(ts) == 0:
            print(f"Warning: Series at index {idx} is empty.")
        elif np.all(np.isnan(ts)):
            print(f"Warning: Series at index {idx} contains only nans.")
    
    # 找到最长的时间序列的长度
    max_len = max(len(ts) for ts in one_label_all_series)

    # 将时间序列填充到相同的长度
    padded_series = [np.pad(ts, (0, max_len - len(ts)), 'constant', constant_values=np.nan) for ts in one_label_all_series]

    # 将列表转换为 (n_ts, sz, 1) 形状的 3D numpy 数组
    X = to_time_series_dataset(padded_series)

    # 数据预处理：将时间序列规范化为零均值和单位方差
    X = TimeSeriesScalerMeanVariance().fit_transform(X)

    # 将3D数组转换为2D，以便我们可以使用SciPy的层次聚类
    X_2d = np.squeeze(X, axis=2)

    # 计算 DTW 距离矩阵
    dist_matrix = cdist_dtw(X)
    # print(dist_matrix)
    # 你的距离矩阵
    dist_matrix = np.array([[0, 2, np.inf], [2, 0, 3], [np.inf, 3, 0]])

    # 替换 dist_matrix 中的 inf 值为 10000
    dist_matrix[np.isinf(dist_matrix)] = 10000

    # Perform hierarchical clustering
    linkage_matrix = linkage(dist_matrix, method='single', optimal_ordering=True, metric='euclidean')

    # Get cluster labels
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    # 找到包含最多时间序列的簇的标签
    (unique, counts) = np.unique(labels, return_counts=True)
    most_common_label = unique[np.argmax(counts)]

    # 选择这个簇中的时间序列
    selected_series = [s for s, l in zip(one_label_all_series, labels) if l == most_common_label]

    # 根据时间序列的长度排序
    sorted_series = sorted(selected_series, key=len)

    # 选择中位数长度的时间序列作为参考
    reference_sequence = sorted_series[len(sorted_series) // 2]
    
    return reference_sequence

def event_detect(signal, pen=2):
    # 在平均时间序列上进行事件检测
    diff_signal = np.diff(signal)
    algo = rpt.Pelt(model="rbf").fit(diff_signal)
    events = algo.predict(pen=pen) # brake小一点比如0.5，最好能够指定一个数量
    return events

def get_event_template(reference_sequence_1, all_series_1, pen):
    """event_template_1 = get_event_template(reference_sequence_1, all_series_1, pen)"""
    # 使用 DTW 对齐所有时间序列，并保存 DTW 路径
    aligned_series = []
    dtw_paths = []
    for idx, series in enumerate(all_series_1):  
        _, path = fastdtw(reference_sequence_1, series, dist=euclidean)
        dtw_paths.append(path)
        aligned = np.zeros_like(reference_sequence_1)
        for i, j in path:
            aligned[i] = series[j]
        aligned_series.append(aligned)

    # 计算平均时间序列
    average_series_1 = np.mean(aligned_series, axis=0)
 
    events = event_detect(average_series_1, pen)
    return events, average_series_1

def align_series(target_length, series):
    """
    Aligns the series to the target length by truncating or padding it.
    
    Parameters:
        target_length (int): The desired length of the series.
        series (pd.Series or np.ndarray): The series to be aligned.
        
    Returns:
        pd.Series or np.ndarray: The aligned series.
    """
    # Convert numpy array to pandas Series for consistent handling
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    
    if len(series) > target_length:
        # If the series is too long, truncate it from the front.
        aligned_series = series[-target_length:]
    elif len(series) < target_length:
        # If the series is too short, pad it with the first value.
        padding = pd.Series([series.iloc[0]] * (target_length - len(series)))
        aligned_series = pd.concat([padding, series], ignore_index=True)
    else:
        # If the series is already the desired length, return it as is.
        aligned_series = series
    
    return aligned_series

# Example usage:
target_length = 100  # Replace with your desired target length
average_series_1 = np.array([1, 2, 3, 4, 5])  # Replace with your actual series

# Now, you should be able to assign it to a column in your DataFrame without encountering a ValueError.
average_series_1_aligned = align_series(target_length, average_series_1)

def fuse_and_map(all_series, average_series_1, event_template_1, ids_to_check, feature, rpt_pen):
    """mapped_events, prior_list = fuse_and_map(all_series, average_series_1, event_template_1, ids_to_check, feature, rpt_pen)"""
    dtw_paths = []
    for idx, series in enumerate(all_series):  
        _, path = fastdtw(average_series_1, series, dist=euclidean)
        dtw_paths.append(path)
   # 使用指数映射计算置信度
    distances = [fastdtw(average_series_1, ts, dist=euclidean)[0] for ts in all_series]
    distances = np.array(distances)
    temperature = np.max(distances)
    prior_list = np.exp(-distances / temperature)

    # 将检测到的事件映射回原始时间序列
    original_events = []

    for path in dtw_paths:
        mapped_events_for_this_path = []
        ref_to_mapped = {}
        
        for ref, mapped in path:
            if ref not in ref_to_mapped:
                ref_to_mapped[ref] = []
            ref_to_mapped[ref].append(mapped)
        
        min_distance = 5  # 选择一个适当的最小距离

        for idx, ref in enumerate(event_template_1):
            if ref in ref_to_mapped:
                if idx == len(event_template_1) - 1:  # 如果是最后一个事件
                    # 映射到原始序列的最后一个时间戳
                    mapped_events_for_this_path.append(path[-1][1])
                elif len(ref_to_mapped[ref]) == 1:
                    # 只有一个映射点
                    mapped_events_for_this_path.append(ref_to_mapped[ref][0])
                else:
                    # 多个映射点，尝试选择一个不太靠近其他已映射点的点
                    chosen_point = None
                    for point in ref_to_mapped[ref]:
                        if all(abs(point - other_point) >= min_distance for other_point in mapped_events_for_this_path):
                            chosen_point = point
                            break
                    if chosen_point is None:
                        # 如果没有找到合适的点，选择中间的点
                        chosen_point = ref_to_mapped[ref][len(ref_to_mapped[ref]) // 2]
                    mapped_events_for_this_path.append(chosen_point)

        original_events.append(mapped_events_for_this_path)

    # 随机选择4个id
    ids_to_plot = random.sample(ids_to_check, 4)
    # ids_to_plot = [30,38,46,64]

    # 获取这4个id对应的序列
    selected_series = [all_series[ids_to_check.index(i)] for i in ids_to_plot]
    selected_original_events = [original_events[ids_to_check.index(i)] for i in ids_to_plot]

    # 设置图形大小
    fig, axes = plt.subplots(len(ids_to_plot) + 1, 1, figsize=(15, 7))

    # 绘制融合的时间序列及其事件
    axes[0].plot(average_series_1, label='Fused Series')
    for event in event_template_1:
        axes[0].axvline(event, color='red', linestyle='--', label='Breakpoint')
    axes[0].set_title("Fused Series with "+str(feature))
    axes[0].legend()

    # 绘制选定的原始序列及其映射事件
    for idx, (series, ax) in enumerate(zip(selected_series, axes[1:])):
        ax.plot(series, label='Original Series')
        for event in selected_original_events[idx]:
            ax.axvline(event, color='red', linestyle='--', label='Mapped-back Breakpoint')
        ax.set_title(f"ID: {ids_to_plot[idx]}")
        ax.legend()

    plt.tight_layout()
    plt.show()

    likelihood = [1]*len(event_template_1)
    prior = 1
    attributes_df = event_attributs(average_series_1, event_template_1, prior, likelihood, feature)
    # 由于你提到event_template_1的最后一个值不需要，我们可以去掉它
    event_starts = event_template_1[:-1]

    # 使用正则表达式来匹配事件名和编号
    pattern = re.compile(r"([\w]+_\d+)")

    # 获取 attributes_df 中的所有独特的事件名（例如：'brake_1', 'brake_2', ...）
    event_names = set(match.group(1) for col in attributes_df.columns for match in [pattern.match(col)] if match)

    # 为每个事件名和每个开始时间添加一个新列
    for event_name in event_names:
        # 获取事件编号
        event_number = event_name.split('_')[-1]
        
        # 确保事件编号是一个数字，并且对应的开始时间存在
        if event_number.isdigit() and int(event_number) <= len(event_starts):
            # 添加新列
            attributes_df[f'{event_name}_start'] = event_starts[int(event_number) - 1]
    # print(attributes_df.columns)
    # 现在，attributes_df 将包含每个事件的 start 列
    # 你可以将 attributes_df 保存为一个 CSV 文件
    path_avg = root_path+str(GPT_iter)+'average_sequence.csv'
    path_att = root_path+str(GPT_iter)+'as_attributes.csv'
    
    # 如果文件存在，读取现有数据
    if os.path.exists(path_avg):
        df_average = pd.read_csv(path_avg)
    else:
        df_average = pd.DataFrame()
    
    if os.path.exists(path_att):
        df_attributes = pd.read_csv(path_att)
    else:
        df_attributes = pd.DataFrame()

    target_length = len(df_average)
    average_series_1_aligned = align_series(target_length, average_series_1)
    # Now, you should be able to assign it to a column in your DataFrame without encountering a ValueError.
    df_average[feature] = average_series_1_aligned
    
    # 找到所有包含 'confidence' 的列
    cols_to_drop = [col for col in attributes_df.columns if 'confidence' in col]
    # 从 attributes_df 中删除这些列
    attributes_df_dropped = attributes_df.drop(columns=cols_to_drop)
    # 然后连接 df_attributes 和 attributes_df_dropped
    df_attributes = pd.concat([df_attributes, attributes_df_dropped], axis=1)

    
    # 存储到 CSV 文件
    df_average.to_csv(path_avg, index=False)
    df_attributes.to_csv(path_att, index=False)

    return original_events, prior_list


def process_mapped_events(series, mapped_events, original_events, temperature=5):
    """
    处理映射事件点，只保留与原生事件点最接近的映射事件点，并计算第二个置信度。
    
    参数：
    - mapped_events: 映射事件点的列表
    - original_events: 原生事件点的列表
    - temperature: 用于计算置信度的温度参数
    
    返回：
    - selected_events: 选定的事件点的列表
    - confidences: 对应于选定事件点的置信度列表
    """
    if len(original_events) == 0 or len(mapped_events) == 0:
        return [], []
    
    if original_events[-1] != len(series)-1:
        original_events.append(len(series)-1)
    if mapped_events[-1] != len(series)-1:
        mapped_events.append(len(series)-1)

    store_original = original_events[-1]
    mapped_events, original_events = mapped_events[:-1], original_events[:-1]

    if len(original_events) <= len(mapped_events):
        return original_events + [store_original], [0.2] * (len(original_events) + 1)
    
    selected_events = []
    likelihood_list = []
    
    for mapped_event in mapped_events:
        # 找到最近的原生事件点
        nearest_event = min(original_events, key=lambda x: abs(x - mapped_event))
        selected_events.append(nearest_event)
        
        # 计算置信度
        distance = abs(nearest_event - mapped_event)
        confidence = np.exp(-distance / temperature)
        likelihood_list.append(confidence)
        
        # 从original_events中移除已匹配的事件，避免重复匹配
        original_events.remove(nearest_event)  
    
    # 处理store_original
    selected_events.append(store_original)
    # 对于store_original，置信度是啥都无所谓，给个1
    likelihood_list.append(1.0)
    return selected_events, likelihood_list


def event_attributs(signal, result, prior, likelihood, feature, color='b'):
    tmp = pd.DataFrame()
    # 如果result为空，添加一行NaN
    if len(result)==0 or 1:
        # print("!!!!!")
        tmp = tmp.append(pd.Series([np.nan] * len(tmp.columns), index=tmp.columns), ignore_index=True)
        # print(tmp)

    for i in range(1, len(result)):
        start, end = result[i-1], result[i]
        change_rate = (signal[end-1] - signal[start]) / (end - start)
        change_magnitude = signal[end-1] - signal[start]
        mean = np.mean(signal[start:end])
        tmp[f'{feature}_{i}_rate'] = [change_rate]
        tmp[f'{feature}_{i}_magnitude'] = [change_magnitude]
        tmp[f'{feature}_{i}_mean'] = [mean]
        tmp[f'{feature}_{i}_start'] = [start]
        tmp[f'{feature}_{i}_confidence'] = [prior*likelihood[i-1]]     
        # TODO: 有的算出来是nan可以替换成0
    return tmp


def plot_events(tmp, axis, signal, feature, result):
    """
    使用tmp DataFrame中的信息绘制事件。

    参数：
    - tmp: DataFrame 包含事件的属性。
    - axis: matplotlib的轴对象 用于绘图。
    - signal: 一维数组或列表，表示信号序列。
    - feature: 字符串，表示特征的名称。
    - result: 列表，包含事件的开始和结束位置。
    """
    # 绘制信号
    axis.plot(signal, label='Signal')
    
    # 遍历result中的每个事件位置
    for i in range(1, len(result)):
        start, end = result[i-1], result[i]

        # 从tmp中提取信息
        change_rate = tmp[f'{feature}_{i}_rate'].values[0]
        change_magnitude = tmp[f'{feature}_{i}_magnitude'].values[0]
        mean = tmp[f'{feature}_{i}_mean'].values[0]
        start = tmp[f'{feature}_{i}_start'].values[0]
        confidence = tmp[f'{feature}_{i}_confidence'].values[0]

        # 标注在图上
        annotation_text = f"Rate: {change_rate:.2f}\nMagnitude: {change_magnitude:.2f}\nMean: {mean:.2f}\nConfidence: {confidence:.2f}"
        axis.annotate(annotation_text, xy=(start, signal[start]),
                      xytext=(0, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"),
                      fontsize=12, ha='center')
        axis.axvline(x=start, linestyle='--', alpha=0.5)  # 在start点画竖线
        axis.axvline(x=end, linestyle='--', alpha=0.5)    # 在end点画竖线

    axis.legend()


def process_one_feature(df, y_df, feature):
    feature, rpt_pen, cluster_thres = feature_set[0], float(feature_set[1]), float(feature_set[2])

    # 单独提取1的label的ids，因为1的数据要作为事件模板，映射回所有原序列
    ids_1 = (y_df[y_df['0'] == 1].index + 1).tolist()
    ids_all = (y_df.index + 1).tolist()

    # 先用聚类得到label=1的事件模板
    all_series_1 = [df[df['id'] == i][feature].values for i in ids_1]
    reference_sequence_1 = hierarchical_clustering_and_reference(all_series_1, threshold=cluster_thres)
    event_template_1, average_series_1 = get_event_template(reference_sequence_1, all_series_1, rpt_pen)

    # 用1的模板映射回0和1的原序列
    all_series = [df[df['id'] == i][feature].values for i in ids_all]
    mapped_events, prior_list = fuse_and_map(all_series, average_series_1, event_template_1, ids_all, feature, rpt_pen)
    original_events = [event_detect(series, rpt_pen*0.4) for series in all_series] #原序列多检测一些断点，用来选
    
    # 创建一个新的DataFrame来存储特征
    df_feature = pd.DataFrame()

    # 处理所有样本，更新df_feature
    for idx, series in enumerate(all_series):
        filter_list, likelihood = process_mapped_events(all_series[idx], mapped_events[idx], original_events[idx])
        prior = prior_list[idx]  # 获取对应的置信度值
        tmp = event_attributs(series, filter_list, prior, likelihood, feature)
        # 获取当前id对应的label
        current_label = y_df.loc[idx, '0']
        
        # 将label添加到tmp DataFrame中
        tmp['result'] = current_label
        
        # 将tmp DataFrame添加到总的DataFrame中
        df_feature = pd.concat([df_feature, tmp], ignore_index=True)
    # print(df_feature)

    # 随机选择4个id进行可视化
    ids_to_check = random.sample(ids_all, 4)

    # 获取这4个id对应的序列
    selected_series = [all_series[ids_all.index(i)] for i in ids_to_check]

    # 获取这4个id对应的映射事件和原始事件
    selected_mapped_events = [mapped_events[ids_all.index(i)] for i in ids_to_check]
    selected_original_events = [original_events[ids_all.index(i)] for i in ids_to_check]

    # 设置图形大小
    fig, axes = plt.subplots(len(ids_to_check), 1, figsize=(15, 7))

    # 使用上述函数进行可视化
    for idx, axis in enumerate(axes):
        filter_list, likelihood = process_mapped_events(selected_series[idx], selected_mapped_events[idx], selected_original_events[idx])
        prior = prior_list[ids_all.index(ids_to_check[idx])]  # 获取对应的置信度值
        
        # 获取事件的属性并存储在tmp DataFrame中
        tmp = event_attributs(selected_series[idx], filter_list, prior, likelihood, feature)
        
        # 使用tmp DataFrame中的信息绘制事件
        plot_events(tmp, axis, selected_series[idx], feature, filter_list)
        
        # 获取当前id对应的label
        current_label = y_df.loc[ids_all.index(ids_to_check[idx]), '0']
        
        # 在图表标题中添加label信息
        # axis.set_title(f"ID: {ids_to_check[idx]}, Label: {current_label}")
        axis.set_title(f"ID: {ids_to_check[idx]}, Label: {current_label}", loc='left')

    plt.tight_layout()
    plt.show()
    return df_feature

def compute_and_fill_features(df, feature_definitions):
    def fill_series(s):
        """用就近的值填充序列中的nan和inf。"""
        if not isinstance(s, pd.Series):
            s = pd.Series(s)
        return s.interpolate(method='nearest').bfill().ffill() 
    for feature_name, feature_definition in feature_definitions.items():
        # 计算新特征
        df[feature_name] = eval(feature_definition)
        
        # 填充 nan 和 inf
        df[feature_name] = fill_series(df[feature_name])
    return df

# 序列对齐
#TODO: 多选几个作为参考序列再取平均，or可以选序列长度中位数的作为参考序列（的长度）

# 读取数据
### 1.改下路径！！！检查好x和y数量是对齐的
root_path = 'assets/1npccutin/'
# root_path = 'assets/data0/'
df = pd.read_csv(root_path+'data_processed.csv')
# y_df = pd.read_csv(root_path+'data_processed_y.csv')
y_df = pd.read_csv(root_path+'features.csv')
y_df = y_df[['id', 'result']]
y_df = y_df.rename(columns={'id': ''})
y_df = y_df.rename(columns={'result': '0'})

### 2.第几次迭代，按GPT的feature公式输入
# 定义你的特征
feature_definitions = {
    'RelativeHeadingNPC1' : "abs(df['heading'] - df['NPC1Theta'])",
}
### !!!!!!!!!!!!
GPT_iter = 1
df = compute_and_fill_features(df, feature_definitions)
# print(df['RelativeDistanceNPC1'])
# ('brake', 0.5, 10),('speed_difference', 0.5, 10),('npc_theta', 2, 5),('relative_distance', 1, 10)
### 3.敏感数据用0.5,10；反之1.5,8
df_proceed = pd.DataFrame()
feature_list = [('NPC1Theta',2.5, 8)] 
# feature_list = [('brake', 0.5, 10),('npc_theta', 2, 5)] 

for i, feature_set in enumerate(feature_list):
    df_feature = process_one_feature(df, y_df, feature_set)
    # 如果不是第一次迭代，并且新的特征数据框包含 "result" 列，则删除它
    if i != 0 and 'result' in df_feature.columns:
        df_feature = df_feature.drop(columns=['result'])
    # 如果是第一次迭代，直接将 df_feature 赋值给 df_proceed
    if i == 0:
        df_proceed = df_feature
    else:
        # 确保你的原始数据框和新的特征数据框有相同的索引
        df_feature = df_feature.set_index(df_proceed.index)
        # 将新的特征数据框添加到处理后的数据框的后面
        df_proceed = pd.concat([df_proceed, df_feature], axis=1)

df_proceed.to_csv(root_path+str(GPT_iter)+'event_extracted.csv', index=False)
