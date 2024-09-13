# DigantaraProject
This is an Assessment for Data Science Intern in Digantara

Concise summary about the methods used in my code:

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

