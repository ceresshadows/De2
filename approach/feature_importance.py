import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import re
from sklearn.inspection import permutation_importance

### Change the path
root_path='assets/case1-lane_borrow/'
GPT_iter = 1

# Read data
df = pd.read_csv(root_path+str(GPT_iter)+'event_extracted.csv')

# Extract and save the Confidence column
confidence_columns = [col for col in df.columns if 'confidence' in col] # Gets the names of all confidence columns
confidence_values = df[confidence_columns].copy() # Save the confidence column values
# print(confidence_values)

# Expand the Confidence column
expanded_confidence = pd.DataFrame()
for conf_col in confidence_columns:   
    # Gets the name of the feature column associated with the confidence column
    feature_prefix = conf_col.rsplit('_', 1)[0]
    # Use regular expressions to find all columns that match the prefix
    related_features = [col for col in df.columns if re.match(fr'{feature_prefix}(_|$)', col) and col != conf_col]
    expanded_confidence[related_features] = pd.DataFrame(np.repeat(confidence_values[[conf_col]].values,  len(related_features), axis=1), columns=related_features)

# Delete the Result and Confidence columns
y = df['result']
X = df.drop(columns=confidence_columns + ['result']) # Delete the confidence and result columns

# Calculate the NaN ratio
nan_ratio_per_feature = X.isna().mean(axis=0) # Calculates the NaN ratio for each feature

# Update Confidence
for feature, nan_ratio in zip(X.columns, nan_ratio_per_feature):
    expanded_confidence[feature] = expanded_confidence[feature] * (1 - nan_ratio)  # update confidence

# Data preprocessing: Fill NaN with the median
X.fillna(X.median(), inplace=True)

noise_scale = 0.1
# Generate white noise with the same length as the dataframe
white_noise = np.random.normal(loc=0.0, scale=noise_scale, size=len(X))
# Add the white noise to the dataframe
X['noise'] = white_noise
expanded_confidence['noise'] = 0.7

### Determine the X and y datasets (do you want to add static features?)
features_df = pd.read_csv(root_path+'features.csv')
static_feature_names = ['rain', 'fog', 'wetness', 'damage', 'daytime']
selected_features = features_df[static_feature_names]
X = pd.concat([X, selected_features], axis=1)

# Divide the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Get feature importance
feature_importances = clf.feature_importances_

# Get the feature name
feature_names = list(X.columns)
# Pair feature names and feature importance and convert them to a dictionary
feature_importance_dict = dict(zip(feature_names, feature_importances))
noise_value = feature_importance_dict['noise']
# Importance of saving newly added features separately
new_feature_importances = {k: feature_importance_dict.pop(k) for k in static_feature_names}
# Update feature_importances to remove the importance of newly added features
feature_importances = np.array(list(feature_importance_dict.values()))

# Calculate the Feature Importance of weighting
# Here we use the average confidence of each feature to calculate the weighted feature importance.
mean_confidence_per_feature = expanded_confidence.mean(axis=0).values
weighted_feature_importances = feature_importances * mean_confidence_per_feature
# print(mean_confidence_per_feature)


# Add the value of new_feature_importances back to the weighted_feature_importances array
weighted_feature_importances = np.append(
    weighted_feature_importances, 
    [value * 0.7 for value in new_feature_importances.values()])
# Normalization Feature Importance
normalized_weighted_feature_importances = weighted_feature_importances / np.sum(weighted_feature_importances)

# Create a data box for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': normalized_weighted_feature_importances
})

# Sort importance_df in descending order of 'Total Importance'
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
# importance_df = importance_df.head(3) # Get the first three most important features

# Select the top three most important features
selected_features = importance_df.head(3)

# Add 'Noise' feature
selected_features = selected_features.append(importance_df[importance_df['Feature'] == 'noise'])

fig, ax = plt.subplots(figsize=(10, 3))

# Create a bar chart
bars = ax.barh(selected_features['Feature'], selected_features['Importance'], color='skyblue')
ax.set_yticks(range(len(selected_features)))
ax.set_yticklabels(selected_features['Feature'])

ax.set_xlabel('Importance')
ax.set_title('First there most important features with noise as reference')
ax.text(0, 2.5, '///', ha='center', va='center', fontsize=20, color='red')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


def ordinal_number(n):
    """Converts numbers to ordinal words"""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def format_feature_name(feature):
    """Formatted feature name"""
    # Match the three parts of the feature name using a regular expression
    match = re.match(r"([a-zA-Z]+)_(\d+)_(\w+)", feature)
    
    if match:
        feature_segment, number, attribute = match.groups()
        
        ordinal_number = f"{number}th" if 10 <= int(number) % 100 <= 20 else \
                         f"{number}{'st' if number.endswith('1') else 'nd' if number.endswith('2') else 'rd' if number.endswith('3') else 'th'}"
        
        formatted_feature_name = f"the {attribute} of {ordinal_number} {feature_segment} segment"
        return formatted_feature_name
    else:
        # Return the original feature name if it does not match the expected format
        return feature


# Make sure importance_df is listed in descending order in the 'Total Importance' column
importance_df = importance_df.sort_values(by='Importance', ascending=False)
most_important_row = importance_df.iloc[0]
feature_expression = f"{format_feature_name(most_important_row['Feature'])} with {most_important_row['Importance']*100:.2f}%"
print(f"Most important feature is {feature_expression} (noise for reference: {noise_value*100:.2f}%)")


def keep_important_features(df, important_feature):
    """ Keep all columns related to the most important features """
    # Extract the main part of the feature (for example, extract "RelativeAngle_2" from "RelativeAngle_2_rate")
    main_feature_part = "_".join(important_feature.split("_")[:-1])
    # Select all columns related to the main feature section
    columns_to_keep = [col for col in df.columns if col == important_feature or col == main_feature_part+"_start"]
    # Keep the selected column
    df_filtered = df[columns_to_keep]
    return df_filtered

df_attributes = pd.read_csv(root_path+str(GPT_iter)+'as_attributes.csv')

# Keep all columns related to the most important features
df_filtered_attributes = keep_important_features(df_attributes, most_important_row['Feature'])

# Store the updated DataFrame
df_filtered_attributes.to_csv(root_path+str(GPT_iter)+'as_attributes_fi.csv', index=False)