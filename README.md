 Assessment Project for the Data Science Intern Role at Digantara:

---

# Digantara Project: Maneuver Detection using Machine Learning

## Overview

This project demonstrates a methodology for detecting maneuvers in orbital data using a Random Forest Classifier. It involves data preprocessing, feature extraction, model training, maneuver detection, and results visualization.

## Methods Used (Why ML? or Heuristic)

### 1. Machine Learning
- **Random Forest Classifier:**
  - **Purpose:** Detect maneuvers based on features extracted from the data.
  - **Why:** Combines multiple decision trees to enhance accuracy and robustness in classification tasks.
  - **How:** Trained on features such as `SMA`, `SMA_diff`, and `SMA_diff_diff` to predict labels (`predicted_label`).

### 2. Heuristic
- **Synthetic Labeling:**
  - **Purpose:** Illustrative labels are created for model training when actual labeled data is unavailable.
  - **Why:** Demonstrates the model's training and evaluation. Actual labeled data should be used in practice.
  - **How:** Labels are assigned using a rule where `SMA_diff_diff > 0.1` is considered a maneuver (label = 1).

## Assumptions

1. **Presence of Label Column:** Assumes the dataset has a `'true_label'` column for training. Without this, the model cannot be trained properly.
2. **Synthetic Labeling for Illustration:** Uses synthetic labels (`SMA_diff_diff > 0.1`) for demonstration. Real labeled data is needed for actual training.
3. **Data Format and Integrity:** Assumes the dataset is in CSV format with specific columns (`'Datetime'`, `'SMA'`, etc.), and data conversion/calculations do not introduce errors.
4. **Machine Learning Model Suitability:** Assumes a Random Forest Classifier is suitable for the task and that the features used are relevant.
5. **Data Distribution and Splitting:** Assumes the data split (using seed 42) is representative of the overall distribution.

## Methodology

### 1. Data Preprocessing
- **Loading Data:** Data is loaded from a CSV file. The `'Datetime'` column is converted to datetime format, and separate `'Date'` and `'Time'` columns are created.
- **Feature Engineering:** Computes features like `SMA_diff` (difference of SMA) and `SMA_diff_diff` (second difference of SMA).

### 2. Feature Extraction
- **Features:** Extracts `SMA`, `SMA_diff`, and `SMA_diff_diff` into a DataFrame with an initial placeholder for labels.

### 3. Model Training
- **Synthetic Label Creation:** Uses a threshold (`SMA_diff_diff > 0.1`) to create synthetic labels for demonstration.
- **Model Building:** Trains a Random Forest Classifier with these features and synthetic labels. Evaluates performance using a classification report.

### 4. Maneuver Detection
- **Prediction:** Applies the trained model to predict labels on the entire dataset.
- **Extraction:** Extracts and lists dates of detected maneuvers.

### 5. Results Visualization
- **Plotting:** Visualizes results by plotting SMA values over time and highlighting detected maneuvers.

## Results

- **Classification Report:** Provides performance metrics such as precision, recall, and F1-score.
- **Detected Maneuvers:** Lists dates where maneuvers are detected, showcasing where significant changes were identified.

## Analysis

- **Model Performance:** Evaluates accuracy and suggests that adjustments to thresholds or model parameters could enhance performance.
- **Visual Insights:** Provides visual confirmation of detected maneuvers and helps validate their occurrence.

## Conclusion

The methodology offers a structured approach for detecting maneuvers using machine learning. Although synthetic labels were used here, the method can be adapted to use real labeled data for practical applications.

## Python Code

For the complete code used in this project, please refer to the `useML.py` file.

---



