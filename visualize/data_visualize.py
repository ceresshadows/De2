import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### read data
timeseries = pd.read_csv('assets/1npccutin/data_processed.csv', index_col=0)

# Get all unique ids
unique_ids = timeseries['id'].unique()
print(len(unique_ids))
selected_ids = np.random.choice(unique_ids, size=15, replace=False)

plt.figure(figsize=(10, 10))

# Draw a trace for each selected id
for uid in selected_ids:
    # Extract the data for the current id
    subset = timeseries.query(f'id=={uid}')

    # Draw ego data
    plt.plot(subset['egoX'], subset['egoY'], label=f'Ego ID: {uid}')

    # Draw the npc's data
    plt.plot(subset['NPC1X'], subset['NPC1Y'], linestyle='dashed', label=f'NPC ID: {uid}')

plt.title('Ego vs NPC')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)

plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取数据
# timeseries = pd.read_csv('assets/for_tsfresh_data.csv', index_col=0)
# y_values = pd.read_csv('assets/for_tsfresh_y.csv', index_col=0)

# plt.figure(figsize=(12, 6))

# # 获取所有的唯一id
# unique_ids = timeseries['id'].unique()

# # 对于每一个id，在同一张图上绘制npc_offset，并根据y值分配颜色
# for uid in unique_ids:
#     subset = timeseries.query(f'id=={uid}')['npc_offset'].reset_index(drop=True)
#     color = 'blue' if y_values.loc[uid]['0'] == 0 else 'red'
#     subset.plot(color=color, alpha=0.5)

# plt.title('NPC Offset for All IDs')
# plt.xlabel('Time Index')
# plt.ylabel('NPC Offset')
# plt.legend(['y=0', 'y=1'], loc='upper right')
# plt.show()
