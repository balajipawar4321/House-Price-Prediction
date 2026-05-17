import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("property_data_with_area.csv")

# -------------------------------
# BASIC CLEANING
# -------------------------------
df.drop_duplicates(inplace=True)
df = df[df["area_sqft"] > 300]
df = df[df["price_per_sqft"] > 2000]

# -------------------------------
# ENCODING
# -------------------------------
area_encoder = LabelEncoder()
df["area"] = area_encoder.fit_transform(df["area"])

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df.drop(columns=["price_per_sqft"])
y = df["price_per_sqft"]

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

print("R2 Score:", round(r2_score(y_test, y_pred), 3))
print("MAE (₹/sqft):", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE (₹/sqft):", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

# -------------------------------
# SAVE MODEL & ENCODER
# -------------------------------
with open("property_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("area_encoder.pkl", "wb") as f:
    pickle.dump(area_encoder, f)

print("✅ Model and encoder saved successfully")
