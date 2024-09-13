# digantaraproject
This is an Assessment for Data Science Intern in Digantara

Here's a concise summary for a presentation slide about the methods used in your code:

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

