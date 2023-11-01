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
root_path='assets/1npccutin/'
GPT_iter =4 

# 读取数据
df = pd.read_csv(root_path+str(GPT_iter)+'event_extracted.csv')

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
static_feature_names = ['rain', 'fog', 'wetness', 'noise', 'damage', 'daytime']
selected_features = features_df[static_feature_names]
X = pd.concat([X, selected_features], axis=1)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 获取特征重要性
feature_importances = clf.feature_importances_

# 获取特征名
feature_names = list(X.columns)
# 将特征名和特征重要性配对并转换为字典
feature_importance_dict = dict(zip(feature_names, feature_importances))
noise_value = feature_importance_dict['noise']
# 单独保存新添加的特征的重要性
new_feature_importances = {k: feature_importance_dict.pop(k) for k in static_feature_names}
# 更新 feature_importances，删除新添加的特征的重要性
feature_importances = np.array(list(feature_importance_dict.values()))


# 计算加权的 Feature Importance
# 这里我们使用每个特征的平均 confidence 来计算加权的 feature importance。
mean_confidence_per_feature = expanded_confidence.mean(axis=0).values
weighted_feature_importances = feature_importances * mean_confidence_per_feature
# print(mean_confidence_per_feature)


# 将 new_feature_importances 的值添加回 weighted_feature_importances 数组中
weighted_feature_importances = np.append(
    weighted_feature_importances, 
    [value * 0.7 for value in new_feature_importances.values()])
# 归一化 Feature Importance
normalized_weighted_feature_importances = weighted_feature_importances / np.sum(weighted_feature_importances)

# 创建一个用于可视化的数据框
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': normalized_weighted_feature_importances
})

# 按 'Total Importance' 降序排序 importance_df
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
# importance_df = importance_df.head(3)  # 获取前三个最重要的特征

# 选择前三个最重要的特征
selected_features = importance_df.head(3)

# 添加 'Noise' 特征
selected_features = selected_features.append(importance_df[importance_df['Feature'] == 'noise'])

fig, ax = plt.subplots(figsize=(10, 3))

# 创建条形图
bars = ax.barh(selected_features['Feature'], selected_features['Importance'], color='skyblue')

# 添加断裂符号
ax.set_yticks(range(len(selected_features)))
ax.set_yticklabels(selected_features['Feature'])

ax.set_xlabel('Importance')
ax.set_title('First there most important features with noise as reference')

# 在特定位置添加断裂符号
ax.text(0, 2.5, '///', ha='center', va='center', fontsize=20, color='red')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


def ordinal_number(n):
    """将数字转换为序数词"""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def format_feature_name(feature):
    """格式化特征名称"""
    # 使用正则表达式匹配特征名称的三个部分
    match = re.match(r"([a-zA-Z]+)_(\d+)_(\w+)", feature)
    
    if match:
        feature_segment, number, attribute = match.groups()
        
        # 将数字转换为序数
        ordinal_number = f"{number}th" if 10 <= int(number) % 100 <= 20 else \
                         f"{number}{'st' if number.endswith('1') else 'nd' if number.endswith('2') else 'rd' if number.endswith('3') else 'th'}"
        
        # 格式化并返回特征名称
        formatted_feature_name = f"the {attribute} of {ordinal_number} {feature_segment} segment"
        return formatted_feature_name
    else:
        # 如果没有匹配到预期的格式，返回原始特征名称
        return feature


# 确保 importance_df 是按 'Total Importance' 列降序排列的
importance_df = importance_df.sort_values(by='Importance', ascending=False)
most_important_row = importance_df.iloc[0]
feature_expression = f"{format_feature_name(most_important_row['Feature'])} with {most_important_row['Importance']*100:.2f}%"
print(f"Most important feature is {feature_expression} (noise for reference: {noise_value*100:.2f}%)")


def keep_important_features(df, important_feature):
    """保留与最重要特征相关的所有列"""
    # 提取特征的主要部分（例如，从 "RelativeAngle_2_rate" 中提取 "RelativeAngle_2"）
    main_feature_part = "_".join(important_feature.split("_")[:-1]) 
    # 选择与主要特征部分相关的所有列
    columns_to_keep = [col for col in df.columns if col == important_feature or col == main_feature_part+"_start"]
    # 保留选定的列
    df_filtered = df[columns_to_keep]
    return df_filtered


df_attributes = pd.read_csv(root_path+str(GPT_iter)+'as_attributes.csv')

# 保留与最重要特征相关的所有列
df_filtered_attributes = keep_important_features(df_attributes, most_important_row['Feature'])

# 存储更新后的 DataFrame
df_filtered_attributes.to_csv(root_path+str(GPT_iter)+'as_attributes_fi.csv', index=False)