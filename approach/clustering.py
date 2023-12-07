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
    
    # Find the length of the longest time series
    max_len = max(len(ts) for ts in one_label_all_series)

    # Fill the time series to the same length
    padded_series = [np.pad(ts, (0, max_len - len(ts)), 'constant', constant_values=np.nan) for ts in one_label_all_series]

    # Converts the list to a 3D numpy array of the shape (n_ts, sz, 1)
    X = to_time_series_dataset(padded_series)

    # Data preprocessing: Normalizes the time series to zero mean and unit variance
    X = TimeSeriesScalerMeanVariance().fit_transform(X)

    # Convert 3D arrays to 2D so that we can use SciPy's hierarchical clustering
    X_2d = np.squeeze(X, axis=2)

    # Calculate the DTW distance matrix
    dist_matrix = cdist_dtw(X)
    # print(dist_matrix)
    # Your distance matrix
    dist_matrix = np.array([[0, 2, np.inf], [2, 0, 3], [np.inf, 3, 0]])

    # Replace the inf value in dist_matrix with 10000
    dist_matrix[np.isinf(dist_matrix)] = 10000

    # Perform hierarchical clustering
    linkage_matrix = linkage(dist_matrix, method='single', optimal_ordering=True, metric='euclidean')

    # Get cluster labels
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    # Find the tag containing the cluster with the most time series
    (unique, counts) = np.unique(labels, return_counts=True)
    most_common_label = unique[np.argmax(counts)]

    # Select the time series in this cluster
    selected_series = [s for s, l in zip(one_label_all_series, labels) if l == most_common_label]

    # Sort by length of time series
    sorted_series = sorted(selected_series, key=len)

    # Select a time series of median length for reference
    reference_sequence = sorted_series[len(sorted_series) // 2]

    return reference_sequence

def event_detect(signal, pen=2):
    # Event detection on average time series
    diff_signal = np.diff(signal)
    algo = rpt.Pelt(model="rbf").fit(diff_signal)
    events = algo.predict(pen=pen) 
    return events

def get_event_template(reference_sequence_1, all_series_1, pen):
    """event_template_1 = get_event_template(reference_sequence_1, all_series_1, pen)"""
    # Use DTW to align all time series and save the DTW path
    aligned_series = []
    dtw_paths = []
    for idx, series in enumerate(all_series_1):  
        _, path = fastdtw(reference_sequence_1, series, dist=euclidean)
        dtw_paths.append(path)
        aligned = np.zeros_like(reference_sequence_1)
        for i, j in path:
            aligned[i] = series[j]
        aligned_series.append(aligned)

    # Calculate the average time series
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
    # Confidence is calculated using an exponential map
    distances = [fastdtw(average_series_1, ts, dist=euclidean)[0] for ts in all_series]
    distances = np.array(distances)
    temperature = np.max(distances)
    prior_list = np.exp(-distances / temperature)

    # Map the detected event back to the original time series
    original_events = []

    for path in dtw_paths:
        mapped_events_for_this_path = []
        ref_to_mapped = {}
        
        for ref, mapped in path:
            if ref not in ref_to_mapped:
                ref_to_mapped[ref] = []
            ref_to_mapped[ref].append(mapped)
        
        min_distance = 5  # Select an appropriate minimum distance

        for idx, ref in enumerate(event_template_1):
            if ref in ref_to_mapped:
                if idx == len(event_template_1) - 1:  # If this is the last event
                    # The last timestamp mapped to the original sequence
                    mapped_events_for_this_path.append(path[-1][1])
                elif len(ref_to_mapped[ref]) == 1:
                    # Only one mapping point
                    mapped_events_for_this_path.append(ref_to_mapped[ref][0])
                else:
                    # Multiple mapped points, try to select a point that is not too close to the other mapped points
                    chosen_point = None
                    for point in ref_to_mapped[ref]:
                        if all(abs(point - other_point) >= min_distance for other_point in mapped_events_for_this_path):
                            chosen_point = point
                            break
                    if chosen_point is None:
                        # If no suitable point is found, select the middle point
                        chosen_point = ref_to_mapped[ref][len(ref_to_mapped[ref]) // 2]
                    mapped_events_for_this_path.append(chosen_point)

        original_events.append(mapped_events_for_this_path)

    # Randomly select 4 ids to draw and see the effect
    ids_to_plot = random.sample(ids_to_check, 4)

    # Get the sequence corresponding to the 4 ids
    selected_series = [all_series[ids_to_check.index(i)] for i in ids_to_plot]
    selected_original_events = [original_events[ids_to_check.index(i)] for i in ids_to_plot]

    # Set the size of the graph
    fig, axes = plt.subplots(len(ids_to_plot) + 1, 1, figsize=(15, 7))

    # Plot the time series of fusion and its events
    axes[0].plot(average_series_1, label='Fused Series')
    for event in event_template_1:
        axes[0].axvline(event, color='red', linestyle='--', label='Breakpoint')
    axes[0].set_title("Fused Series with "+str(feature))
    axes[0].legend()

    # Draw the selected original sequence and its mapping events
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
    # The last value of event_template is not needed, we can remove it
    event_starts = event_template_1[:-1]

    # 使用正则表达式来匹配事件名和编号
    pattern = re.compile(r"([\w]+_\d+)")

    # 获取 attributes_df 中的所有独特的事件名（例如：'brake_1', 'brake_2', ...）
    event_names = set(match.group(1) for col in attributes_df.columns for match in [pattern.match(col)] if match)

    # Add a new column for each event name and each start time
    for event_name in event_names:
        # Get the event number
        event_number = event_name.split('_')[-1]
        
        # Ensure that the event number is a number and that the corresponding start time exists
        if event_number.isdigit() and int(event_number) <= len(event_starts):
            attributes_df[f'{event_name}_start'] = event_starts[int(event_number) - 1]
    path_avg = root_path+str(GPT_iter)+'average_sequence.csv'
    path_att = root_path+str(GPT_iter)+'as_attributes.csv'
    
    # If file exists, read existing data
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
    
    # Find all columns containing 'confidence'
    cols_to_drop = [col for col in attributes_df.columns if 'confidence' in col]
    # Remove these columns from attributes_df
    attributes_df_dropped = attributes_df.drop(columns=cols_to_drop)
    # Then connect df_attributes and attributes_df_dropped
    df_attributes = pd.concat([df_attributes, attributes_df_dropped], axis=1)


    # Save to CSV file
    df_average.to_csv(path_avg, index=False)
    df_attributes.to_csv(path_att, index=False)

    return original_events, prior_list


def process_mapped_events(series, mapped_events, original_events, temperature=5):
    """
    The mapped event points are processed, keeping only the mapped event points closest to the native event points, and the second confidence level is calculated.

    Parameters:
    - mapped_events: indicates the list of mapped event points
    - original_events: indicates the original_events list
    - temperature: Temperature parameter used to calculate confidence

    Back:
    - selected_events: Lists the selected event points
    - confidences: Confidence lists corresponding to selected event points
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
        # Find the nearest native event point
        nearest_event = min(original_events, key=lambda x: abs(x - mapped_event))
        selected_events.append(nearest_event)
        
        # Calculate confidence
        distance = abs(nearest_event - mapped_event)
        confidence = np.exp(-distance / temperature)
        likelihood_list.append(confidence)
        
        # Removes matched events from original_events to avoid duplicate matches
        original_events.remove(nearest_event)  
    
    # Handle store_original
    selected_events.append(store_original)
    # For store_original, it doesn't matter what the confidence is, give it a 1
    likelihood_list.append(1.0)
    return selected_events, likelihood_list


def event_attributs(signal, result, prior, likelihood, feature, color='b'):
    tmp = pd.DataFrame()
    # If result is empty, add a line of NaN
    if len(result)==0 or 1:
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
    return tmp


def plot_events(tmp, axis, signal, feature, result):
    """
    Plot events using the information in the tmp DataFrame.

    Parameters:
    -tmp: DataFrame contains the attribute of the event.
    - axis: The axis object of matplotlib is used for drawing.
    - signal: indicates a one-dimensional array or list, indicating a signal sequence.
    - feature: A character string indicating the feature name.
    - result: A list containing the start and end locations of events.
    """
    # Draw signal
    axis.plot(signal, label='Signal')
    
    # Iterate over each event location in result
    for i in range(1, len(result)):
        start, end = result[i-1], result[i]

        # Extract information from tmp
        change_rate = tmp[f'{feature}_{i}_rate'].values[0]
        change_magnitude = tmp[f'{feature}_{i}_magnitude'].values[0]
        mean = tmp[f'{feature}_{i}_mean'].values[0]
        start = tmp[f'{feature}_{i}_start'].values[0]
        confidence = tmp[f'{feature}_{i}_confidence'].values[0]

        # Label on the picture
        annotation_text = f"Rate: {change_rate:.2f}\nMagnitude: {change_magnitude:.2f}\nMean: {mean:.2f}\nConfidence: {confidence:.2f}"
        axis.annotate(annotation_text, xy=(start, signal[start]),
                      xytext=(0, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"),
                      fontsize=12, ha='center')
        axis.axvline(x=start, linestyle='--', alpha=0.5)  # Draw a vertical line on the start dot
        axis.axvline(x=end, linestyle='--', alpha=0.5)    # Draw a vertical line at the end

    axis.legend()


def process_one_feature(df, y_df, feature):
    feature, rpt_pen, cluster_thres = feature_set[0], float(feature_set[1]), float(feature_set[2])

    # Extract the label ids of 1 separately
    # because the data of 1 is to be used as the event template and mapped back to all the original sequences
    ids_1 = (y_df[y_df['0'] == 1].index + 1).tolist()
    ids_all = (y_df.index + 1).tolist()

    # First get the event template with label=1 by clustering
    all_series_1 = [df[df['id'] == i][feature].values for i in ids_1]
    reference_sequence_1 = hierarchical_clustering_and_reference(all_series_1, threshold=cluster_thres)
    event_template_1, average_series_1 = get_event_template(reference_sequence_1, all_series_1, rpt_pen)

    # Map back to the original sequence of 0 and 1 using the template of 1
    all_series = [df[df['id'] == i][feature].values for i in ids_all]
    mapped_events, prior_list = fuse_and_map(all_series, average_series_1, event_template_1, ids_all, feature, rpt_pen)
    # The original sequence detects some breakpoints, used to select
    original_events = [event_detect(series, rpt_pen*0.4) for series in all_series] 
    
    # Create a new DataFrame to store the feature
    df_feature = pd.DataFrame()

    # Process all samples and update df_feature
    for idx, series in enumerate(all_series):
        filter_list, likelihood = process_mapped_events(all_series[idx], mapped_events[idx], original_events[idx])
        prior = prior_list[idx]  # Gets the corresponding confidence value
        tmp = event_attributs(series, filter_list, prior, likelihood, feature)
        # Gets the label corresponding to the current id
        current_label = y_df.loc[idx, '0']
        
        # Add label to tmp DataFrame
        tmp['result'] = current_label
        
        # Add tmp DataFrame to the total DataFrame
        df_feature = pd.concat([df_feature, tmp], ignore_index=True)
    # print(df_feature)

    # Randomly select 4 ids for visualization
    ids_to_check = random.sample(ids_all, 4)

    # Get the sequence corresponding to the 4 ids
    selected_series = [all_series[ids_all.index(i)] for i in ids_to_check]

    # Get the mapped event and the original event corresponding to the 4 ids
    selected_mapped_events = [mapped_events[ids_all.index(i)] for i in ids_to_check]
    selected_original_events = [original_events[ids_all.index(i)] for i in ids_to_check]

    # Set the size of the graph
    fig, axes = plt.subplots(len(ids_to_check), 1, figsize=(15, 7))

    # Use the above functions for visualization
    for idx, axis in enumerate(axes):
        filter_list, likelihood = process_mapped_events(selected_series[idx], selected_mapped_events[idx], selected_original_events[idx])
        prior = prior_list[ids_all.index(ids_to_check[idx])]  # Gets the corresponding confidence value
        
        # Gets the attributes of the event and stores them in the tmp DataFrame
        tmp = event_attributs(selected_series[idx], filter_list, prior, likelihood, feature)

        # Draw events using the information in the tmp DataFrame
        plot_events(tmp, axis, selected_series[idx], feature, filter_list)

        # Gets the label corresponding to the current id
        current_label = y_df.loc[ids_all.index(ids_to_check[idx]), '0']

        # Add label information to the chart title
        # axis.set_title(f"ID: {ids_to_check[idx]}, Label: {current_label}")
        axis.set_title(f"ID: {ids_to_check[idx]}, Label: {current_label}", loc='left')

    plt.tight_layout()
    plt.show()
    return df_feature

def compute_and_fill_features(df, feature_definitions):
    def fill_series(s):
        """ Fills nan and inf in the sequence with the nearest value."""
        if not isinstance(s, pd.Series):
            s = pd.Series(s)
        return s.interpolate(method='nearest').bfill().ffill() 
    for feature_name, feature_definition in feature_definitions.items():
        # Calculate new features
        df[feature_name] = eval(feature_definition)
        
        # Fill nan and inf
        df[feature_name] = fill_series(df[feature_name])
    return df

# Read data
### 1.edit root_path
root_path = 'assets/1npccutin/'

df = pd.read_csv(root_path+'data_processed.csv')

y_df = pd.read_csv(root_path+'features.csv')
y_df = y_df[['id', 'result']]
y_df = y_df.rename(columns={'id': ''})
y_df = y_df.rename(columns={'result': '0'})

### 2.interface with LLM agent
feature_definitions = {
    'LaneChange' : "abs(df['heading'] - df['NPC1Theta'])",
}
GPT_iter = 1
df = compute_and_fill_features(df, feature_definitions)

### 3.Use 0.5,10 for sensitive data; Vice versa 1.5,8
df_proceed = pd.DataFrame()
feature_list = [('NPC1Theta',2.5, 8)] 

for i, feature_set in enumerate(feature_list):
    df_feature = process_one_feature(df, y_df, feature_set)
    # If it is not the first iteration and the new feature data box contains the "result" column, delete it
    if i != 0 and 'result' in df_feature.columns:
        df_feature = df_feature.drop(columns=['result'])
    # If this is the first iteration, assign df_feature directly to df_proceed
    if i == 0:
        df_proceed = df_feature
    else:
        # Make sure your original data box and the new feature data box have the same index
        df_feature = df_feature.set_index(df_proceed.index)
        # Add the new feature data box to the end of the processed data box
        df_proceed = pd.concat([df_proceed, df_feature], axis=1)

df_proceed.to_csv(root_path+str(GPT_iter)+'event_extracted.csv', index=False)
