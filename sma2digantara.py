
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'SMA_data.csv'
sma_data = pd.read_csv(file_path)

# Preprocessing
sma_data['Datetime'] = pd.to_datetime(sma_data['Datetime'])
sma_data['Date'] = sma_data['Datetime'].dt.date
sma_data = sma_data[['Date', 'SMA']]

# Calculate SMA differences
sma_data['SMA_diff'] = sma_data['SMA'].diff()

# Function to calculate maneuvers based on a given threshold
def detect_maneuvers(threshold):
    sma_data['Maneuver'] = np.abs(sma_data['SMA_diff']) > threshold
    return sma_data

# Grid search for optimal threshold
thresholds = np.arange(3,8.1,5.92)
best_threshold = None
best_count = float('inf')

for t in thresholds:
    threshold = sma_data['SMA_diff'].mean() + (sma_data['SMA_diff'].std() * t)
    detected_data = detect_maneuvers(threshold)
    maneuver_count = detected_data['Maneuver'].sum()
    
    if maneuver_count < best_count:
        best_count = maneuver_count
        best_threshold = threshold

# Print the optimal threshold
print(f"Optimal Threshold: {best_threshold}")

# Detect maneuvers using the optimal threshold
sma_data = detect_maneuvers(best_threshold)
maneuver_data = sma_data[sma_data['Maneuver']]

# Display and plot results
print("Detected Maneuvers:")
print(maneuver_data[['Date', 'SMA']])

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