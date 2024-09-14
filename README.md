
Assessment Project for the Data Science Intern Role at Digantara:


---

# Digantara Project: Maneuver Detection using Machine Learning

## Overview

This project utilizes machine learning to detect maneuvers in orbital data with a Random Forest Classifier. The workflow includes data preprocessing, feature extraction, model training, maneuver detection, and result visualization. The aim is to identify significant changes in orbital data that may signify maneuvers.

## Why Machine Learning?

### Advantages

- **Adaptability:** Captures complex patterns beyond simple rules.
- **Robustness:** Uses ensemble methods (Random Forest) to minimize overfitting.
- **Feature Interaction:** Automatically assesses feature importance and interactions.
- **Quantitative Metrics:** Provides performance metrics like precision, recall, and F1-score.
- **Scalability:** Models can be updated with new data to adapt to evolving patterns.

### Heuristic Approach

The demonstration uses synthetic labels (`SMA_diff_diff > 0.1`). For practical applications, real labeled data should be used.

## Methodology

1. **Data Preprocessing:**
   - Load data from CSV.
   - Convert `Datetime` to datetime format; extract `Date` and `Month`.
   - Drop the original `Datetime` column.

2. **Feature Extraction:**
   - Compute additional features: `SMA_diff` and `SMA_diff_diff`.
   - Initialize placeholders for labels.

3. **Model Training:**
   - Generate synthetic labels for training.
   - Train a Random Forest Classifier and evaluate it.
   - Predict maneuvers across the dataset.

4. **Maneuver Detection:**
   - Extract and list dates with detected maneuvers (one per month).

5. **Results Visualization:**
   - Plot SMA values and detected maneuvers.
   - Save results as PNG files.

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/kiranraoboinapally/digantaraproject.git
   cd digantaraproject
   ```

2. **Install Dependencies:**

   Ensure Python 3.x is installed. Install the required libraries:

   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

3. **Prepare Your Data:**

   - Ensure your data is in `SMA_data.csv` with columns including `Datetime`, `SMA`, etc.
   - Place the file in the root directory of the project.

4. **Run the Script:**

   Execute the script to preprocess data, train the model, and generate visualizations:

   ```bash
   python useML.py
   ```

5. **Review Results:**

   - **`graphOutput.png`**: A plot showing SMA values and detected maneuvers.
   - **`datesOutput.png`**: A PNG file containing a table of detected maneuver dates.

## Code Overview

The main script `useML.py` includes:

- **Data Preprocessing:** Functions for loading and preparing data.
- **Feature Extraction:** Functions for calculating `SMA_diff` and `SMA_diff_diff`.
- **Model Training:** Training and evaluating the Random Forest Classifier.
- **Maneuver Detection:** Extracting and listing detected maneuvers.
- **Visualization:** Plotting SMA values and maneuver detection results, and saving them as PNG files.

### `useML.py` Script

- **`load_data(file_path)`**: Loads and preprocesses the data.
- **`preprocess_data(df)`**: Computes additional SMA features.
- **`extract_features(df)`**: Prepares features for model training.
- **`train_model(features, df)`**: Trains the Random Forest model and predicts maneuvers.
- **`extract_maneuver_dates(df, features)`**: Extracts dates where maneuvers are detected.
- **`plot_results(df, features)`**: Plots SMA values and saves the plot to `graphOutput.png`.
- **`save_maneuver_dates(maneuver_dates)`**: Saves detected maneuver dates to `datesOutput.png`.

## Assumptions

- **Data Format:** Having CSV file with columns including `Datetime` and `SMA`.
- **Label Column:** Uses synthetic labels for demonstration; real data is preferred for actual use.
- **Model Choice:** Random Forest Classifier is assumed suitable for the task.

## Future Work

- **Real Data Integration:** Use actual labeled data for training and evaluation.
- **Model Optimization:** Explore hyperparameter tuning and alternative models.
- **Feature Enhancement:** Investigate additional features and data sources.

## Contact

For questions or support, please contact [kiranraoboinapally@gmail.com](mailto:kiranraoboinapally@gmail.com).


## Acknowledgments

- **Data Source:** The dataset used in this project is a synthetic dataset provided as part of the assignment. It is used here for illustrative purposes only.

- **Libraries:** This project utilizes several key libraries:
  - **Pandas**: For data manipulation and analysis. [Pandas Documentation](https://pandas.pydata.org/)
  - **NumPy**: For numerical operations and handling arrays. [NumPy Documentation](https://numpy.org/)
  - **scikit-learn**: For machine learning algorithms and model evaluation. [scikit-learn Documentation](https://scikit-learn.org/)
  - **Matplotlib**: For data visualization and plotting. [Matplotlib Documentation](https://matplotlib.org/)


---

 



