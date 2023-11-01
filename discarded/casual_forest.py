import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import re
from sklearn.inspection import permutation_importance

### 改路径
root_path='assets/1failtoyield/'

# 读取数据
df = pd.read_csv(root_path+'event_extracted.csv')

# 假设 df 是你的数据框

# 提取和保存 Confidence 列
confidence_columns = [col for col in df.columns if 'confidence' in col]  # 获取所有 confidence 列的名称
confidence_values = df[confidence_columns].copy()  # 保存 confidence 列的值
# print(confidence_values)

# 扩展 Confidence 列
expanded_confidence = pd.DataFrame()
for conf_col in confidence_columns:   
    # 获取与 confidence 列关联的特征列的名称
    feature_prefix = conf_col.rsplit('_', 1)[0]     
    # 使用正则表达式来查找与前缀匹配的所有列
    related_features = [col for col in df.columns if re.match(fr'{feature_prefix}(_|$)', col) and col != conf_col]  
    expanded_confidence[related_features] = pd.DataFrame(np.repeat(confidence_values[[conf_col]].values, len(related_features), axis=1), columns=related_features)


# 删除 Result 列和 Confidence 列
y = df['result']
X = df.drop(columns=confidence_columns + ['result'])  # 删除 confidence 列和 result 列

# 计算 NaN 比例
nan_ratio_per_feature = X.isna().mean(axis=0)  # 计算每个特征的 NaN 比例

# 更新 Confidence
for feature, nan_ratio in zip(X.columns, nan_ratio_per_feature):
    expanded_confidence[feature] = expanded_confidence[feature] * (1 - nan_ratio)  # 更新 confidence
# print(expanded_confidence.columns)

# 数据预处理：使用中位数填充 NaN
X.fillna(X.median(), inplace=True)

### 确定好X和y数据集（是否要加静态特征！！）
features_df = pd.read_csv(root_path+'features.csv')
selected_features = features_df[['rain', 'fog', 'wetness', 'cloudiness', 'damage', 'Time_of_day']]
X = pd.concat([X, selected_features], axis=1)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 获取特征重要性
feature_importances = clf.feature_importances_


# 获取特征名
feature_names = X.columns
# 将特征名和特征重要性配对并转换为字典
feature_importance_dict = dict(zip(feature_names, feature_importances))
# 单独保存新添加的特征的重要性
new_feature_importances = {k: feature_importance_dict.pop(k) for k in ['rain', 'fog', 'wetness', 'cloudiness', 'damage', 'Time_of_day']}
# 更新 feature_importances，删除新添加的特征的重要性
feature_importances = np.array(list(feature_importance_dict.values()))


# 计算加权的 Feature Importance
# 这里我们使用每个特征的平均 confidence 来计算加权的 feature importance。
mean_confidence_per_feature = expanded_confidence.mean(axis=0).values
weighted_feature_importances = feature_importances * mean_confidence_per_feature
# print(mean_confidence_per_feature)


# 将 new_feature_importances 的值添加回 weighted_feature_importances 数组中
weighted_feature_importances = np.append(weighted_feature_importances, list(new_feature_importances.values()))
# 归一化 Feature Importance
normalized_weighted_feature_importances = weighted_feature_importances / np.sum(weighted_feature_importances)
# 现在，normalized_weighted_feature_importances 包含所有特征（包括新添加的特征）的归一化重要性

# 创建一个字典来存储每个事件的总重要性
event_importance = {}

# 遍历所有的特征和它们的重
for feature, importance in zip(X.columns, normalized_weighted_feature_importances):
    # 提取事件名称
    match = re.match(r"([a-zA-Z_]+_\d+)", feature)
    if match:
        event_name = match.group(1)
        # 将特征的重要性添加到事件的总重要性中
        event_importance[event_name] = event_importance.get(event_name, 0) + importance
    else:
        event_importance[feature] = importance
# 读取现有的数据
df_attributes = pd.read_csv(root_path+'average_sequence_attributes.csv')

# 添加新的列
for event_name, importance in event_importance.items():
    df_attributes[f'{event_name}_importance'] = importance

# 存储更新后的 DataFrame
df_attributes.to_csv(root_path+'average_sequence_attributes.csv', index=False)

# 创建一个用于可视化的数据框
importance_df = pd.DataFrame(list(event_importance.items()), columns=['Event', 'Total Importance'])

# 可视化
plt.figure(figsize=(10, 4))
plt.barh(importance_df['Event'], importance_df['Total Importance'], color='skyblue')
plt.xlabel('Total Feature Importance')
plt.ylabel('Event')
plt.title('Event Importance Visualization')
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()

# 按 'Total Importance' 降序排序 importance_df
importance_df.sort_values(by='Total Importance', ascending=False, inplace=True)
top_features = importance_df.head(3)  # 获取前三个最重要的特征

def ordinal_number(n):
    """将数字转换为序数词"""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def format_feature_name(feature):
    """格式化特征名称"""
    # 提取数字
    number_match = re.search(r'_(\d+)', feature)
    if number_match:
        number = int(number_match.group(1))
        ordinal = ordinal_number(number)
        # 删除数字并用序数词替换
        feature_name_parts = feature.split('_')
        feature = f"{ordinal} {' '.join(feature_name_parts[:-1])} segment"
    else:
        feature = ' '.join(feature.split('_'))
    return feature

# 创建一个字符串，列出最重要的特征及其相对重要性
important_features_str = ', '.join([f"{format_feature_name(row['Event'])} ({row['Total Importance']*100:.0f}%)" 
                                   for _, row in top_features.iterrows()])

# 输出结果
print(f"Most important features: {important_features_str}")

