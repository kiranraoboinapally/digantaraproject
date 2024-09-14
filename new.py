import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# Import and register matplotlib converters
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Define the output directory path
output_dir = os.path.join(os.path.dirname(__file__), 'output')

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Data Preprocessing
def load_data(file_path):
    """
    Load the data from a CSV file and preprocess it.
    
    Parameters:
    file_path (str): Path to the CSV file containing the data.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(file_path)
    print(df.columns)  # Print column names for verification
    
    # Convert the DateTime column to datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    
    # Create separate Date and Month columns
    df['Date'] = df['Datetime'].dt.date
    df['Month'] = df['Datetime'].dt.to_period('M')  # Add a Month column for grouping
    
    # Drop the original DateTime column
    df.drop(['Datetime'], axis=1, inplace=True)
    
    return df

def preprocess_data(df):
    """
    Compute additional features related to SMA differences.
    
    Parameters:
    df (pd.DataFrame): DataFrame with SMA column.
    
    Returns:
    pd.DataFrame: DataFrame with additional features.
    """
    df['SMA_diff'] = df['SMA'].diff().fillna(0)
    df['SMA_diff_diff'] = df['SMA_diff'].diff().fillna(0)
    return df

# Feature Extraction
def extract_features(df):
    """
    Extract features for model training.
    
    Parameters:
    df (pd.DataFrame): DataFrame with preprocessed data.
    
    Returns:
    pd.DataFrame: DataFrame with features and placeholder labels.
    """
    features = pd.DataFrame()
    features['SMA'] = df['SMA']
    features['SMA_diff'] = df['SMA_diff']
    features['SMA_diff_diff'] = df['SMA_diff_diff']
    
    # Placeholder for labels (to be filled with synthetic labels)
    features['label'] = np.nan
    
    return features

# Maneuver Detection with Machine Learning
def train_model(features, df):
    """
    Train a Random Forest Classifier to detect maneuvers.
    
    Parameters:
    features (pd.DataFrame): DataFrame with features and synthetic labels.
    df (pd.DataFrame): DataFrame with original data.
    
    Returns:
    pd.DataFrame: DataFrame with predicted labels.
    """
    # Generate synthetic labels for demonstration purposes
    features['label'] = np.where(features['SMA_diff_diff'] > 0.1, 1, 0)
    
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
    """
    Extract dates where maneuvers are detected.
    
    Parameters:
    df (pd.DataFrame): DataFrame with original data.
    features (pd.DataFrame): DataFrame with predicted labels.
    
    Returns:
    pd.DataFrame: DataFrame with dates of detected maneuvers.
    """
    # Filter rows where maneuvers are detected (predicted_label == 1)
    maneuvers = df[features['predicted_label'] == 1]
    
    # Keep only one maneuver per month
    maneuvers = maneuvers.drop_duplicates(subset='Month', keep='first')
    
    return maneuvers[['Date']]

# Plot Results
def plot_results(df, features):
    """
    Plot the SMA values and highlight detected maneuvers.
    
    Parameters:
    df (pd.DataFrame): DataFrame with original data.
    features (pd.DataFrame): DataFrame with predicted labels.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot the SMA values
    plt.plot(df['Date'], df['SMA'], label='SMA', color='blue')
    
    # Highlight the maneuvers
    maneuvers = df[features['predicted_label'] == 1].drop_duplicates(subset='Month', keep='first')
    plt.scatter(maneuvers['Date'], maneuvers['SMA'], color='red', label='Detected Maneuvers', zorder=5)
    
    plt.xlabel('Date')
    plt.ylabel('SMA')
    plt.title('Detected Maneuvers in Orbital Data')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to file in the output directory
    plt.savefig(os.path.join(output_dir, 'graphOutput.png'))

def save_maneuver_dates(maneuver_dates):
    """
    Save the detected maneuver dates to a PNG file.
    
    Parameters:
    maneuver_dates (pd.DataFrame): DataFrame with dates of detected maneuvers.
    """
    plt.figure(figsize=(10, 6))
    plt.table(cellText=maneuver_dates.values,
              colLabels=maneuver_dates.columns,
              cellLoc='center',
              loc='center',
              edges='closed')
    plt.axis('off')
    
    # Save table to file in the output directory
    plt.savefig(os.path.join(output_dir, 'datesOutput.png'))

# Main function
def main():
    """
    Main function to execute the project workflow.
    """
    file_path = 'SMA_data.csv'  # Replace with the actual file path
    df = load_data(file_path)
    df = preprocess_data(df)
    features = extract_features(df)
    
    features = train_model(features, df)
    
    maneuver_dates = extract_maneuver_dates(df, features)
    print("Dates where maneuvers are detected:")
    print(maneuver_dates)
    
    plot_results(df, features)
    save_maneuver_dates(maneuver_dates)
    
    # Print the final message
    print("Check the output autosaved in the output folder.")

if __name__ == "__main__":
    main()
