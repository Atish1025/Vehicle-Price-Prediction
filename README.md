## PVP - Price Value Prediction
###  Project Overview
**PVP (Price Value Prediction)** is a machine learning project aimed at predicting the resale price of used vehicles based on various vehicle attributes. The model is trained using structured data that includes features like brand, model, mileage, fuel type, transmission, color, and more. This project demonstrates the use of data preprocessing, feature engineering, and regression modeling to create a robust vehicle price prediction system.

---

### 🔍 Problem Statement

With the boom in the used car market, both buyers and sellers are looking for accurate valuation models. Manual estimation can be subjective and unreliable. This project uses a Random Forest Regressor to build a predictive model that provides objective and data-driven vehicle pricing.

---

### 🧰 Technologies Used

* **Python**
* **Pandas, NumPy** – Data manipulation and numerical computation
* **Matplotlib, Seaborn** – Data visualization
* **Scikit-learn** – Machine learning (model building, evaluation, and preprocessing)

---

### ⚙️ Workflow

1. **Importing Libraries**: Required Python libraries are imported for data processing, visualization, and model building.
2. **Loading Dataset**: Reads a dataset containing various vehicle attributes and prices.
3. **Data Cleaning**:

   * Drops irrelevant columns like `description`, `name`, and `engine`.
   * Removes rows with missing values.
4. **Feature Engineering**:

   * Calculates the vehicle age.
   * Removes the `year` column post-calculation.
5. **Categorical Encoding**: Uses one-hot encoding for categorical variables to make the data ML-compatible.
6. **Train-Test Split**: Splits the dataset into 80% training and 20% testing.
7. **Model Training**: Trains a `RandomForestRegressor` on the training set.
8. **Evaluation**: Evaluates the model using Mean Absolute Error (MAE) and R² score.
9. **Feature Importance**: Visualizes the top 10 features impacting price prediction.

---

### 📈 Results

* **MAE** and **R² Score** are printed to assess the performance of the model.
* The model shows strong predictive power and highlights key pricing factors like mileage, age, make, and fuel type.

---

### 📊 Sample Output

* MAE: *Low values indicate high accuracy*
* R² Score: *Closer to 1 means better model fit*
* Feature importance chart helps identify influential attributes.

---

### 📂 Dataset

* Assumes a CSV file named `dataset.csv` with vehicle data.
* Features include:

  * `make`, `model`, `year`, `mileage`, `fuel`, `transmission`, `body`, etc.
  * Target: `price`

---

### 🚀 Future Improvements

* Hyperparameter tuning
* Try advanced models like XGBoost or Gradient Boosting
* Integrate a web dashboard using Streamlit or Flask
* Add support for user-input predictions

---

### 📎 Use Cases

* Used car dealerships
* Car resale platforms
* Price estimators for individual sellers
