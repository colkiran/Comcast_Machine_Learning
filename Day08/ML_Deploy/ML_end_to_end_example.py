# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 2. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Define model + hyperparameter grid
rf = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20]
}

grid = GridSearchCV(rf, param_grid, cv=3, scoring="r2")

# 6. Fit model
grid.fit(X_train_scaled, y_train)

# 7. Evaluate model
y_pred = grid.best_estimator_.predict(X_test_scaled)
print("Best Params:", grid.best_params_)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 8. Save model + scaler into joblib
joblib.dump(grid.best_estimator_, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
