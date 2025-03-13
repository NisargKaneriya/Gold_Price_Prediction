# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# from sklearn.linear_model import LinearRegression, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import r2_score

# # Load the CSV file
# file_path = "data/Gold (2).csv"
# df = pd.read_csv(file_path)

# # Convert Date column to datetime
# df['Date'] = pd.to_datetime(df['Date'])

# # Drop missing values
# df = df.dropna()

# # Feature Engineering
# df["High-Low"] = df["High"] - df["Low"]
# df["Open-Close"] = df["Open"] - df["Close/Last"]

# # Selecting features & target
# X = df[['Open', 'High', 'Low', 'Volume', 'High-Low', 'Open-Close']]
# y = df['Close/Last']

# # Splitting data into training & testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # K-Fold Cross Validation
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# # Dictionary to store results
# model_scores = {}

# ### 1️⃣ Linear Regression
# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)
# y_pred_lr = lr_model.predict(X_test)
# model_scores['Linear Regression'] = r2_score(y_test, y_pred_lr)

# ### 2️⃣ Lasso Regression
# lasso_model = Lasso(alpha=0.1)
# lasso_model.fit(X_train, y_train)
# y_pred_lasso = lasso_model.predict(X_test)
# model_scores['Lasso Regression'] = r2_score(y_test, y_pred_lasso)

# ### 3️⃣ Decision Tree Regressor (Hyperparameter Tuning)
# dt_params = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
# dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_params, cv=kfold)
# dt_grid.fit(X_train, y_train)
# dt_model = dt_grid.best_estimator_
# y_pred_dt = dt_model.predict(X_test)
# model_scores['Decision Tree'] = r2_score(y_test, y_pred_dt)

# ### 4️⃣ Random Forest Regressor (Hyperparameter Tuning)
# rf_params = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
# rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=kfold)
# rf_grid.fit(X_train, y_train)
# rf_model = rf_grid.best_estimator_
# y_pred_rf = rf_model.predict(X_test)
# model_scores['Random Forest'] = r2_score(y_test, y_pred_rf)

# # Print all model accuracies
# print("\n Model Performance (R² Scores):")
# for model, score in model_scores.items():
#     print(f"{model}: {score:.4f}")

# # Determine the best model
# best_model_name = max(model_scores, key=model_scores.get)
# best_model = {
#     "Linear Regression": lr_model,
#     "Lasso Regression": lasso_model,
#     "Decision Tree": dt_model,
#     "Random Forest": rf_model,
# }[best_model_name]

# # Save the best model
# best_model_path = "best_model.pkl"
# with open(best_model_path, "wb") as f:
#     pickle.dump(best_model, f)

# print(f"\n Best Model: {best_model_name} (R² Score: {model_scores[best_model_name]:.4f})")
# print(f" Model saved as '{best_model_path}'")
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# Load Data
file_path = "data/Gold (2).csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Drop missing values
df = df.dropna()

# Feature Engineering
df["High-Low"] = df["High"] - df["Low"]

# Rolling Features (5-day rolling mean and exponential moving average)
df["Rolling_Mean"] = df["Close/Last"].rolling(window=5).mean()
df["EWMA"] = df["Close/Last"].ewm(span=5, adjust=False).mean()

# Fill NaN values caused by rolling window
df = df.fillna(method="bfill")

# Selecting features & target
X = df[['Open', 'High', 'Low', 'Volume', 'High-Low', 'Rolling_Mean', 'EWMA']]
y = df['Close/Last']

# Splitting data into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features (important for SVR, KNN, Gradient Boosting, etc.)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store results
model_scores = {}

### Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
model_scores['Linear Regression'] = r2_score(y_test, lr_model.predict(X_test))

### Lasso Regression
lasso_model = Lasso(alpha=0.01)  # Reduced alpha for better generalization
lasso_model.fit(X_train, y_train)
model_scores['Lasso Regression'] = r2_score(y_test, lasso_model.predict(X_test))

### Decision Tree Regressor (Hyperparameter Optimization)
dt_params = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
dt_grid = RandomizedSearchCV(DecisionTreeRegressor(random_state=42), dt_params, cv=kfold, n_iter=5, random_state=42)
dt_grid.fit(X_train, y_train)
dt_model = dt_grid.best_estimator_
model_scores['Decision Tree'] = r2_score(y_test, dt_model.predict(X_test))

### Random Forest Regressor (Hyperparameter Optimization)
rf_params = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
rf_grid = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=kfold, n_iter=5, random_state=42)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
model_scores['Random Forest'] = r2_score(y_test, rf_model.predict(X_test))

### Gradient Boosting Regressor (Boosted Trees)
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
model_scores['Gradient Boosting'] = r2_score(y_test, gb_model.predict(X_test))

# Print all model R² scores
print("\n Model Performance (R² Scores):")
for model, score in model_scores.items():
    print(f"{model}: {score:.4f}")

# Determine the best model
best_model_name = max(model_scores, key=model_scores.get)
best_model = {
    "Linear Regression": lr_model,
    "Lasso Regression": lasso_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
}[best_model_name]

# Save the best model
best_model_path = "best_model_1.pkl"
with open(best_model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"\n Best Model: {best_model_name} (R² Score: {model_scores[best_model_name]:.4f})")
print(f" Model saved as '{best_model_path}'")
