

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset (replace 'SMA_data.csv' with the path to your actual file)
file_path = 'SMA_data.csv'  # Update the file path accordingly
sma_data = pd.read_csv(file_path)

# Preprocessing: Convert the 'Datetime' column to datetime type
sma_data['Datetime'] = pd.to_datetime(sma_data['Datetime'])

# Separate the 'Datetime' column into 'Date' column only
sma_data['Date'] = sma_data['Datetime'].dt.date

# Reorder columns so 'Date' comes before 'SMA'
sma_data = sma_data[['Date', 'SMA']]

# Calculate the SMA rate of change (difference between consecutive values)
sma_data['SMA_diff'] = sma_data['SMA'].diff()

# Calculate the mean and standard deviation of the SMA differences
mean_sma_diff = sma_data['SMA_diff'].mean()
std_sma_diff = sma_data['SMA_diff'].std()

# Set a threshold as a multiple of the standard deviation (e.g., 3 std deviations)
threshold = mean_sma_diff + (std_sma_diff * 3)

# Detect maneuvers based on threshold
sma_data['Maneuver'] = np.abs(sma_data['SMA_diff']) > threshold

# Extract detected maneuvers
maneuver_data = sma_data[sma_data['Maneuver']]

# Display the detected maneuver dates and corresponding SMA values
maneuver_table = maneuver_data[['Date', 'SMA']]
print("Detected Maneuvers:")
print(maneuver_table)

# Plotting the SMA data and detected maneuvers
plt.figure(figsize=(10, 6))
plt.plot(sma_data['Date'], sma_data['SMA'], label='SMA', color='blue', linewidth=1)
plt.scatter(maneuver_data['Date'], maneuver_data['SMA'], color='red', label='Detected Maneuver', marker='o')
plt.xlabel('Date')
plt.ylabel('SMA')
plt.title('SMA with Detected Maneuvers')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()