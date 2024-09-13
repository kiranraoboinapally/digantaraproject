 Assessment Project for the Data Science Intern Role at Digantara:


---

# Digantara Project: Maneuver Detection using Machine Learning

## Overview

This project demonstrates a methodology for detecting maneuvers in orbital data using a Random Forest Classifier. The approach includes data preprocessing, feature extraction, model training, maneuver detection, and results visualization.

## Methods Used

### 1. Machine Learning

- **Random Forest Classifier:**
  - **Purpose:** To detect maneuvers based on features extracted from the data.
  - **Why:** It enhances accuracy and robustness by combining multiple decision trees, which is particularly useful for complex classification tasks.
  - **How:** The model is trained on features such as `SMA`, `SMA_diff`, and `SMA_diff_diff` to predict maneuver labels (`predicted_label`).

### 2. Heuristic

- **Synthetic Labeling:**
  - **Purpose:** To create illustrative labels for model training when actual labeled data is unavailable.
  - **Why:** Demonstrates model training and evaluation. In practice, actual labeled data should be used.
  - **How:** Labels are generated with a rule where `SMA_diff_diff > 0.1` is considered a maneuver (label = 1).

## Assumptions

1. **Presence of Label Column:** Assumes the dataset has a `'true_label'` column for proper model training. Without this, training is not feasible.
2. **Synthetic Labeling:** Uses synthetic labels for demonstration. Real labeled data is required for actual training and evaluation.
3. **Data Format and Integrity:** Assumes the dataset is in CSV format with columns like `'Datetime'` and `'SMA'`, and that data conversion/calculations are accurate.
4. **Model Suitability:** Assumes the Random Forest Classifier is appropriate for this task and that the chosen features are relevant.
5. **Data Distribution and Splitting:** Assumes the data split (using seed 42) is representative of the overall distribution.

## Methodology

### 1. Data Preprocessing
- **Loading Data:** Load data from a CSV file. Convert the `'Datetime'` column to datetime format and create separate `'Date'` and `'Month'` columns.
- **Feature Engineering:** Compute features like `SMA_diff` (difference of SMA) and `SMA_diff_diff` (second difference of SMA).

### 2. Feature Extraction
- **Features:** Extract `SMA`, `SMA_diff`, and `SMA_diff_diff` into a DataFrame, initially with placeholder labels.

### 3. Model Training
- **Synthetic Label Creation:** Generate synthetic labels using a threshold (`SMA_diff_diff > 0.1`) for demonstration purposes.
- **Model Building:** Train a Random Forest Classifier on these features and synthetic labels. Evaluate the model's performance using a classification report.

### 4. Maneuver Detection
- **Prediction:** Apply the trained model to predict labels across the entire dataset.
- **Extraction:** Identify and list dates of detected maneuvers.

### 5. Results Visualization
- **Plotting:** Create visualizations showing SMA values over time and highlight detected maneuvers.

## Results

- **Classification Report:** Provides metrics such as precision, recall, and F1-score to evaluate model performance.
- **Detected Maneuvers:** Lists dates where maneuvers are detected, showcasing significant changes in the data.

## Analysis

- **Model Performance:** Evaluates accuracy and suggests that adjustments to thresholds or model parameters could enhance performance.
- **Visual Insights:** Provides visual confirmation of detected maneuvers, aiding in the validation of detected events.

## Conclusion

This project provides a structured approach for detecting maneuvers using machine learning. While synthetic labels were used in this demonstration, the methodology is adaptable for practical applications with real labeled data.

## Python Code

For the complete code used in this project, please refer to the `useML.py` file.

---



 



