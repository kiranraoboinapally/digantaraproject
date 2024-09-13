# DigantaraProject
This is an Assessment for Data Science Intern in Digantara

Here's a concise summary about the methods used in my code:

---

### **Methods Used**

**1. Machine Learning:**
   - **Random Forest Classifier:**
     - **Purpose:** Used to detect maneuvers in the data based on features extracted.
     - **Why:** It builds multiple decision trees and aggregates their results to improve accuracy and robustness in classification tasks.
     - **How:** The model is trained on features like `SMA`, `SMA_diff`, and `SMA_diff_diff`, and predicts labels (`predicted_label`).

**2. Heuristic:**
   - **Synthetic Labeling:**
     - **Purpose:** Created for illustrative purposes when actual labeled data is not available.
     - **Why:** Provides a way to demonstrate model training and evaluation. In practice, real labeled data should be used.
     - **How:** Labels are assigned based on a fixed rule where `SMA_diff_diff > 0.1` is classified as a maneuver (label = 1).

---


Here’s how to address each of your requests:

### **1. List of Assumptions Made:**

1. **Presence of Label Column:** The code assumes that the dataset contains a column named `'true_label'` which provides the actual labels needed for training the Random Forest model. If this column does not exist, the model cannot be trained properly.

2. **Synthetic Labeling for Illustration:** The code uses synthetic labels (`SMA_diff_diff > 0.1`) for demonstration purposes. In a real-world scenario, actual labeled data is required for accurate training and evaluation.

3. **Data Format and Integrity:** The code assumes that the dataset is in a CSV format with specific columns (`'Datetime'`, `'SMA'`, etc.). It also assumes that data conversion (e.g., to datetime format) and feature calculations will not introduce significant errors or missing values.

4. **Machine Learning Model Suitability:** It assumes that a Random Forest Classifier is an appropriate choice for the classification task and that the chosen features (`SMA`, `SMA_diff`, `SMA_diff_diff`) are relevant for detecting maneuvers.

5. **Data Distribution and Splitting:** The code splits data into training and test sets using a fixed random seed (42). It assumes this split is representative of the overall data distribution.

### **2. Comprehensive Document**

**Title: Methodology, Results, and Analysis of Maneuver Detection Using Machine Learning**

**Introduction:**
This document summarizes the methodology used for detecting maneuvers in orbital data using a Random Forest Classifier, along with the results and analysis of the approach.

**Methodology:**

1. **Data Preprocessing:**
   - **Loading Data:** Data is loaded from a CSV file. The 'Datetime' column is converted to datetime format, and separate 'Date' and 'Time' columns are created.
   - **Feature Engineering:** Features such as `SMA_diff` (difference of SMA) and `SMA_diff_diff` (second difference of SMA) are computed to capture changes in the SMA values.

2. **Feature Extraction:**
   - Features are extracted into a DataFrame, including `SMA`, `SMA_diff`, and `SMA_diff_diff`. A placeholder for labels is initialized.

3. **Model Training:**
   - **Synthetic Label Creation:** For demonstration purposes, synthetic labels are created where maneuvers are classified based on a fixed threshold (`SMA_diff_diff > 0.1`).
   - **Model Building:** A Random Forest Classifier is trained on the features with the synthetic labels. The model's performance is evaluated using a classification report.

4. **Maneuver Detection:**
   - **Prediction:** The trained model is used to predict labels on the entire dataset, indicating where maneuvers are detected.
   - **Extraction:** Dates of detected maneuvers are extracted for further analysis.

5. **Results Visualization:**
   - **Plotting:** The results are visualized by plotting the SMA values over time and highlighting detected maneuvers.

**Results:**
- **Classification Report:** The Random Forest model's performance metrics include precision, recall, and F1-score, which provide insights into its effectiveness in detecting maneuvers.
- **Detected Maneuvers:** Dates where maneuvers are detected are listed, showcasing where the model identified significant changes.

**Analysis:**
- **Model Performance:** The classification report helps assess the model's accuracy. Adjustments to the synthetic label threshold or model parameters may improve performance.
- **Visual Insights:** The plotted results illustrate the model's predictions over time, helping to visualize detected maneuvers and validate their occurrence.

**Conclusion:**
The methodology provides a structured approach to detecting maneuvers using machine learning. While synthetic labels were used for demonstration, the approach can be adapted with real labeled data for practical applications.

### **3. Python Code**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Data Preprocessing
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.columns)  # Print column names for verification
    
    # Convert the DateTime column to datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    
    # Create separate Date and Time columns
    df['Date'] = df['Datetime'].dt.date
    
    # Drop the original DateTime column
    df.drop(['Datetime'], axis=1, inplace=True)
    
    return df

def preprocess_data(df):
    df['SMA_diff'] = df['SMA'].diff().fillna(0)
    df['SMA_diff_diff'] = df['SMA_diff'].diff().fillna(0)
    return df

# Feature Extraction
def extract_features(df):
    features = pd.DataFrame()
    features['SMA'] = df['SMA']
    features['SMA_diff'] = df['SMA_diff']
    features['SMA_diff_diff'] = df['SMA_diff_diff']
    
    # Placeholder for labels (for training the model)
    features['label'] = np.nan  # Initialize with NaN as we will not use heuristics for labeling
    
    return features

# Maneuver Detection with Machine Learning
def train_model(features, df):
    # In this case, we'll need labeled data to train our model. For this example, 
    # I'll create synthetic labels for the purpose of demonstration.
    # In a real-world scenario, you should use actual labeled data for training.
    features['label'] = np.where(features['SMA_diff_diff'] > 0.1, 1, 0)  # Synthetic label for illustration
    
    X = features[['SMA', 'SMA_diff', 'SMA_diff_diff']]
    y = features['label']
    
    # Drop rows where the label is NaN
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Check for NaN values in features
    if X.isna().sum().sum() > 0:
        print("Warning: There are NaN values in the feature set.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Apply the model to the entire dataset to detect maneuvers
    features['predicted_label'] = model.predict(X[['SMA', 'SMA_diff', 'SMA_diff_diff']])
    return features

# Extract Dates with Detected Maneuvers
def extract_maneuver_dates(df, features):
    # Filter rows where maneuvers are detected (predicted_label == 1)
    maneuver_dates = df[features['predicted_label'] == 1][['Date']]
    return maneuver_dates

# Plot Results
def plot_results(df, features):
    plt.figure(figsize=(14, 7))
    
    # Plot the SMA values
    plt.plot(df['Date'], df['SMA'], label='SMA', color='blue')
    
    # Highlight the maneuvers
    maneuvers = df[features['predicted_label'] == 1]
    plt.scatter(maneuvers['Date'], maneuvers['SMA'], color='red', label='Detected Maneuvers', zorder=5)
    
    plt.xlabel('Date')
    plt.ylabel('SMA')
    plt.title('Detected Maneuvers in Orbital Data')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function
def main():
    file_path = 'SMA_data.csv'  # Replace with the actual file path
    df = load_data(file_path)
    df = preprocess_data(df)
    features = extract_features(df)
    
    features = train_model(features, df)
    
    maneuver_dates = extract_maneuver_dates(df, features)
    print("Dates where maneuvers are detected:")
    print(maneuver_dates)
    
    plot_results(df, features)

if __name__ == "__main__":
    main()
```
