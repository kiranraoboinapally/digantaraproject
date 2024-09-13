 Assessment Project for the Data Science Intern Role at Digantara:

---

# Digantara Project: Maneuver Detection using Machine Learning

## Overview

This project demonstrates how to detect maneuvers in orbital data using a Random Forest Classifier. The methodology involves data preprocessing, feature extraction, model training, maneuver detection, and visualization of results. The goal is to identify significant changes in the data that may indicate maneuvers.

## Why Machine Learning?

### Advantages of Machine Learning (ML)

- **Adaptability:** ML models can capture complex patterns in the data that simple heuristic rules might miss.
- **Robustness:** Random Forest Classifier combines multiple decision trees to enhance model performance and reduce overfitting.
- **Feature Interaction:** ML models can automatically learn and evaluate the importance of different features and their interactions.
- **Quantitative Evaluation:** ML provides performance metrics such as precision, recall, and F1-score, which offer a systematic way to assess model effectiveness.
- **Scalability:** ML models can be retrained with new data, making them more adaptable to changes in data patterns.

### Heuristic Approach

For demonstration purposes, synthetic labels are created using a simple heuristic (`SMA_diff_diff > 0.1`). While useful for illustration, real-world applications should use actual labeled data to train and evaluate models.

## Methodology

1. **Data Preprocessing:**
   - Load data from a CSV file.
   - Convert the `Datetime` column to a datetime format.
   - Create separate `Date` and `Month` columns for analysis.
   - Drop the original `Datetime` column.

2. **Feature Extraction:**
   - Calculate additional features such as `SMA_diff` (difference of SMA) and `SMA_diff_diff` (second difference of SMA).
   - Initialize a placeholder for labels.

3. **Model Training:**
   - Generate synthetic labels for training.
   - Train a Random Forest Classifier with the features and synthetic labels.
   - Evaluate model performance using a classification report.

4. **Maneuver Detection:**
   - Apply the trained model to predict maneuvers across the dataset.
   - Extract dates with detected maneuvers and ensure only one maneuver per month is listed.

5. **Results Visualization:**
   - Plot SMA values over time.
   - Highlight detected maneuvers on the plot.

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/kiranraoboinapally/digantaraproject.git
   cd digantaraproject
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.x installed, and then install the required libraries:

   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

3. **Prepare Your Data:**

   - Ensure your data is in a CSV file named `SMA_data.csv` with columns including `Datetime`, `SMA`, etc.
   - Place this file in the root directory of the project.

4. **Run the Script:**

   Execute the script to perform data preprocessing, model training, and visualization:

   ```bash
   python useML.py
   ```

5. **Review Results:**

   - The script will print a classification report to the console.
   - Detected maneuver dates will be displayed.
   - A plot showing SMA values and detected maneuvers will be generated.

## Code

The main script for this project is `useML.py`. It includes:

- **Data Preprocessing:** Functions for loading and preparing data.
- **Feature Extraction:** Functions for calculating necessary features.
- **Model Training:** Function for training and evaluating the Random Forest model.
- **Maneuver Detection:** Function for extracting and listing detected maneuvers.
- **Visualization:** Function for plotting results.

## Assumptions

1. **Data Format:** Assumes the dataset is in CSV format with appropriate columns.
2. **Label Column:** Assumes availability of a label column; synthetic labels are used here for demonstration.
3. **Model Suitability:** Assumes that the Random Forest Classifier is suitable for the task.




---

 



