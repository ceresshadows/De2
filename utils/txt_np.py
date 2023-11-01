import numpy as np
import pandas as pd
import os

def placeholder_to_np(data, keys):
    """
    Convert placeholders in the specified columns of the dataframe to numpy arrays and return only those columns.
    
    Parameters:
    - data: The dataframe containing placeholders.
    - keys: A list of column names that need to be converted from placeholders to numpy arrays.
    
    Returns:
    - new_data: A dataframe with only the specified columns converted to numpy arrays.
    """
    new_data = pd.DataFrame()
    for key in keys:
        if key in data.columns:
            new_data[key] = data[key].apply(lambda filepath: np.loadtxt(filepath))
    return new_data


def np_to_placeholder(data, keys):
    """
    Convert numpy arrays in the specified columns of the dataframe back to placeholders.
    
    Parameters:
    - data: The dataframe containing numpy arrays.
    - keys: A list of column names that need to be converted from numpy arrays to placeholders.
    - output_dir: The directory where numpy arrays are saved.
    
    Returns:
    - data: The dataframe with placeholders instead of numpy arrays for the specified columns.
    """
    for key in keys:
        if key in data.columns:
            for idx, array in enumerate(data[key]):
                data.at[idx, key] = save_array_to_txt(array, key, idx)
    return data



def save_array_to_txt(array, filename_prefix, row_index):
    """
    Save a numpy array to a text file and return the filename.
    
    Parameters:
    - array: The numpy array to save.
    - filename_prefix: The prefix for the filename.
    - output_dir: The directory to save the file in.
    - row_index: The index of the row in the dataframe.
    
    Returns:
    - filename: The path to the saved file.
    """
    output_dir = 'assets/numpy'
    filename = os.path.join(output_dir, f'{filename_prefix}_{row_index}.txt').replace('\\', '/')
    if isinstance(array[0], (int, float, np.number)):  # Check if the first element is a number
        np.savetxt(filename, array, fmt='%s')  # Save as numbers
    else:
        with open(filename, 'w') as f:
            for item in array:
                f.write("%s\n" % item)  # Save as strings
    return filename
