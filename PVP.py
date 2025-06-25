# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 2. Load Dataset
df = pd.read_csv('dataset.csv')  

# 3. Quick Overview
print(df.head())
print(df.info())

# 4. Drop Unnecessary Columns or Handle Missing Values
df.drop(['description', 'name', 'engine'], axis=1, inplace=True)
df.dropna(inplace=True)

# 5. Feature Engineering
df['age'] = 2025 - df['year']
df.drop('year', axis=1, inplace=True)

# 6. Define Categorical and Numerical Features
cat_features = ['make', 'model', 'fuel', 'transmission', 'trim', 'body', 
                'exterior_color', 'interior_color', 'drivetrain']
num_features = ['cylinders', 'mileage', 'doors', 'age']

# 7. Encode Categorical Variables
df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)

# 8. Prepare Features and Target
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# 9. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 11. Evaluate Model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 12. Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()
