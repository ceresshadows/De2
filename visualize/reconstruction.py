import pandas as pd
import numpy as np
import re
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms

### edit before run
iters_to_process = [1,4]
 
root_path = 'assets/1npccutin/'
description = {}

def key_state_and_icon(iter_to_process):
    global description
    def plot_icon(whole_feature_name, feature_name, attribute_name, key_att):
        fig, ax = plt.subplots(figsize=(2, 2))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        if attribute_name == 'mean':
            ax.axhline(y=0.5, color='r', linewidth=2)
            ax.text(0.5, 0.1, f'the mean of {feature_name} = {key_att:.2f}', horizontalalignment='center', fontsize=15)
            
        elif attribute_name in ['rate', 'magnitude']:
            if key_att > 0:
                y1, y2 = 0.2, 0.8
            else:
                y1, y2 = 0.8, 0.2
            ax.plot([0.2, 0.8], [y1, y2], color='r', linewidth=2)
         
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
            
        file_path = os.path.join(root_path+'mini_plots/', f"{whole_feature_name}.png") 
        plt.savefig(file_path)  


    state_df = pd.read_csv(root_path+str(iter_to_process)+'as_attributes_fi.csv')
    columns = state_df.columns
    if len(columns)==1:
        match = re.match(r"(.+?)_(\d+)_(.+)", columns[0])
        if match:
            feature_name, number, attribute_name = match.groups()           
            whole_feature_name = feature_name+'_'+number+'_'+attribute_name
            key_att = state_df[whole_feature_name].iloc[0]
        else:
            print(f"Not matched: {columns[0]}")
        key_start = state_df[feature_name+'_'+number+'_start'].iloc[0]
        description[key_start+1]=f'The key feature of scenario is the trigger time of {feature_name} that is {key_att:.2f}.' 
        return feature_name+'_'+number, key_att, key_start

    for column in columns:
        if "start" in column:
            continue
        # Match column names with a regular expression, and add a negative lookahead assertion to exclude column names that contain "_start"
        match = re.match(r"(.+?)_(\d+)_(.+)", column)
        if match:
            feature_name, number, attribute_name = match.groups()           
            whole_feature_name = feature_name+'_'+number+'_'+attribute_name
            key_att = state_df[whole_feature_name].iloc[0]
        else:
            print(f"Not matched: {column}")
        key_start = state_df[feature_name+'_'+number+'_start'].iloc[0]
        # Call the function to create the icon
        plot_icon(whole_feature_name, feature_name, attribute_name, key_att)
    description[key_start+1]=f'The key feature of scenario is the {attribute_name} of {feature_name} becomes {key_att:.2f}.' 
    return feature_name+'_'+number, key_att, key_start

def mini_plot_and_delta(start_time, event_name):
    # Separate handling of collision, such as start_time =-1
    df = pd.read_csv(root_path+'data_processed.csv')
    df_y = pd.read_csv(root_path+'features.csv')
    # Filter the data box to contain only rows with result 1
    df_filtered = df_y[df_y['result'] == 1]
    # Select 10 (or as many) unique ids at random from the filtered data box
    ids_to_average = random.sample(list(df_filtered['id'].unique()), min(len(df_filtered), 10))

    ### Whether there are multiple NPCs
    features_to_av = ['egoX', 'egoY', 'heading', 'NPC1X', 'NPC1Y', 'NPC1Theta']
    # The data is further filtered and only the id in ids_to_average is retained
    df_filtered = df_filtered[df_filtered['id'].isin(ids_to_average)]
    # Group the data by id, and then select the row in each group
    if start_time == -1:
        df_filtered = df.groupby('id').apply(lambda x: x.iloc[-1]).dropna()
    df_filtered = df.groupby('id').apply(lambda x: x.iloc[start_time] if len(x) > start_time else None).dropna()
    average_values = {}
    for feature in features_to_av:
        # Calculate the average value of each feature in the filtered data box and store it in the dictionary
        average_values[feature] = np.mean(df_filtered[feature])

    global X_LIM, Y_LIM   
    def plot_car(x, y, angle, car_icon, ax):
        im = ax.imshow(car_icon, extent=[x - 6, x + 6, y - 6, y + 6])
        rotate_transform = mtransforms.Affine2D().rotate_deg_around(x, y, angle*180/np.pi) + ax.transData
        im.set_transform(rotate_transform)
    ego_icon_path = 'assets/plot/ego.png'
    npc_icon_path = 'assets/plot/npc.png'
    
    ego_icon = mpimg.imread(ego_icon_path)
    npc_icon = mpimg.imread(npc_icon_path)

    fig, ax = plt.subplots()
    # If you call the function for the first time, set the axis range based on the vehicle position
    if X_LIM is None or Y_LIM is None:
        X_LIM = (min(average_values['egoX'], average_values['NPC1X']) - 25, 
                 max(average_values['egoX'], average_values['NPC1X']) + 10)
        Y_LIM = (min(average_values['egoY'], average_values['NPC1Y']) - 10, 
                 max(average_values['egoY'], average_values['NPC1Y']) + 25)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)

    plot_car(average_values['egoX'], average_values['egoY'], average_values['heading'], ego_icon, ax)
    plot_car(average_values['NPC1X'], average_values['NPC1Y'], average_values['NPC1Theta'], npc_icon, ax)

    file_path = os.path.join(root_path+'mini_plots/', f"{event_name}.png")  
    plt.savefig(file_path)  

    plt.close(fig)
    global description
    if start_time == -1:
        start_time = 1000
    description[start_time]=f"""The state of current scenario is: deltaX of ego & NPC = {average_values['NPC1X']-average_values['egoX']:.2f}
        deltaY of ego & NPC = {average_values['NPC1Y']-average_values['egoY']:.2f}
        heading of ego = {average_values['heading']:.2f}
        heading of NPC = {average_values['NPC1Theta']:.2f}"""
    return average_values


def action_plot(start, end):
    df = pd.read_csv(root_path+'data_processed.csv')
    df_y = pd.read_csv(root_path+'features.csv')
    df_y_1 = df_y[df_y['result'] == 1]
    df_y_0 = df_y[df_y['result'] == 0]
    # id classification
    ids_1 = df_y_1['id'].unique()
    ids_0 = df_y_0['id'].unique()
    features_to_av = ['throttle', 'brake', 'steering']
    # Filter the data further to keep only items with a specific id
    df_data_1 = df[df['id'].isin(ids_1)]
    df_data_0 = df[df['id'].isin(ids_0)]
    # Calculated sampling point
    if end != -1:
        detect_point = (start + end) // 2
        print(f"End != -1, detect_point set to: {detect_point}")  # Prints the value of detect_point
    else:
        detect_point_func = lambda x: min((start + len(x)) // 2,len(x)-5)
        detect_point = int(df.groupby('id').apply(lambda x: detect_point_func(x)).mean())
        print(f"End == -1, detect_point set to: {detect_point}")  # Prints the value of detect_point

    # Value by sampling point
    df_detected_1 = df_data_1.groupby('id').apply(
        lambda x: x.iloc[detect_point_func(x)] if end == -1 else (x.iloc[detect_point] if len(x) > detect_point else None)
    ).dropna()
    df_detected_0 = df_data_0.groupby('id').apply(
        lambda x: x.iloc[detect_point_func(x)] if end == -1 else (x.iloc[detect_point] if len(x) > detect_point else None)
    ).dropna()
    features_to_av = ['throttle', 'brake', 'steering']  # Draw feature list

    # Create a new graphic
    plt.figure(figsize=(10, 6))
    for i, feature in enumerate(features_to_av):
        # Get data for each feature
        data_0 = df_detected_0[feature].dropna() # Gets the data for group 0 and deletes the NaN value
        data_1 = df_detected_1[feature].dropna() # Get the data for group 1 and delete the NaN value

        # Save the mean of the original data
        mean_0_original = data_0.mean()
        mean_1_original = data_1.mean()

        # Scale data
        max_value = max(data_0.max(), data_1.max())
        min_value = min(data_0.min(), data_1.min())
        range_value = max_value - min_value
        data_0 = (data_0 - min_value) / range_value
        data_1 = (data_1 - min_value) / range_value

        # Create a box diagram
        bplot1 = plt.boxplot(data_0, positions=[i*3], widths=0.6, patch_artist=True,
        boxprops=dict(facecolor='yellow'), medianprops=dict(color='black'), labels=['No Collision']) # Group 0 Green
        bplot2 = plt.boxplot(data_1, positions=[i*3+1], widths=0.6, patch_artist=True,
        boxprops=dict(facecolor='pink'), medianprops=dict(color='black'), labels=['Collision']) # Group 1 red

        # Add the mean tag (using the mean of the raw data)
        plt.text(i*3, data_0.mean(), f'{mean_0_original:.2f}', ha='center', va='center', color='black', fontsize=28)
        plt.text(i*3+1, data_1.mean(), f'{mean_1_original:.2f}', ha='center', va='center', color='black', fontsize=28)

    # Set the label for the X-axis
    plt.xticks(np.arange(len(features_to_av)) * 3 + 0.5, features_to_av, fontsize=28)

    # Add legend
    plt.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ['No Collision', 'Collision'], loc='upper right')
    
    file_path = os.path.join(root_path+'mini_plots/', f"action_{detect_point}.png") 
    plt.savefig(file_path) 

    global description 
    description[detect_point] = f"""The action of ego now is throttle: {df_detected_1["throttle"].mean():.2f}, 
            brake: {df_detected_1["brake"].mean():.2f}, steering: {df_detected_1["steering"].mean():.2f},
            while the action of ego in the absence of NPC would be 
            throttle: {df_detected_1["throttle"].mean():.2f}, brake: {df_detected_0["brake"].mean():.2f}, steering: {df_detected_0["steering"].mean():.2f}"""
    return detect_point

# Define a global variable to store the axis range
X_LIM = None
Y_LIM = None

attributes, starts, event_names =[], [], []
actions = []
# Scene status point
for i in iters_to_process:
    event_name, att, start = key_state_and_icon(i)
    attributes.append(att)
    starts.append(start)
    event_names.append(event_name)

# Operating point
for i in range(len(starts)):
    seg_start = starts[i]
    if i != len(starts)-1:
        seg_end = starts[i+1]
    else:
        seg_end = -1
    actions.append(action_plot(seg_start,seg_end))
# mini_plot_and_delta(50, 'collision')

# initial point, every state point and Collision
for i in range(len(iters_to_process)+2):
    if i ==0: # initial
        mini_plot_and_delta(max(0,starts[0]-18), 'initial')
    elif i ==len(iters_to_process)+1:
        mini_plot_and_delta(-1, 'collision')
    else:
        mini_plot_and_delta(starts[i-1], event_names[i-1])
   
# Key sort
sorted_description = sorted(description.items())

# Print each key-value pair
for key, value in sorted_description:
    if key == 1000:
        print(f'At collision point, {value}')
        break
    print(f'At time {key}: {value}')