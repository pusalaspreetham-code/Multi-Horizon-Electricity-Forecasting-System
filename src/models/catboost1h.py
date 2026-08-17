import gc
import os
import time
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)

# ==========================================================
# SMAPE
# ==========================================================

def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100

# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading dataset...")

df = pd.read_parquet("../data/Forecasting_1h.parquet")

print("Dataset Shape:", df.shape)

# ==========================================================
# TRAIN TEST SPLIT (BY METER)
# ==========================================================

meters = np.sort(df["meter"].unique())

train_meters = meters[:30]
test_meters = meters[30:]

mask = df["meter"].isin(train_meters)

# ==========================================================
# FEATURES
# ==========================================================

drop_columns = [
    "target_1h",          # or target_1m / target_1d based on your dataset
    "meter",
    "x_Timestamp",
    "z_Avg Voltage (Volt)",
    "z_Avg Current (Amp)",
    "y_Freq (Hz)",

    "Quarter",
    "Month",
    "Weekend",
    "WeekOfYear",
    "BusinessHour",
    "DayOfWeek",
    "Season",
    "Month_cos",
    "Month_sin",
    "Precipitation",
    "lag_6",
    "rolling_mean_3",
    "lag_12",
    "lag_3",
    "rolling_mean_72",
    "rolling_mean_168",
    "last_6h_energy",
    "EveningPeak"
]

feature_columns = [
    c for c in df.columns
    if c not in drop_columns
]

print("Number of Features:", len(feature_columns))

# ==========================================================
# PREPARE TRAINING DATA
# ==========================================================

print("Preparing training data...")

X_train = df.loc[mask, feature_columns].to_numpy(dtype=np.float32)
y_train = df.loc[mask, "target_1h"].to_numpy(dtype=np.float32)

print("Preparing testing data...")

X_test = df.loc[~mask, feature_columns].to_numpy(dtype=np.float32)
y_test = df.loc[~mask, "target_1h"].to_numpy(dtype=np.float32)

gc.collect()

print("Train Shape :", X_train.shape)
print("Test Shape  :", X_test.shape)

# ==========================================================
# CATBOOST MODEL
# ==========================================================

model = CatBoostRegressor(

    iterations=1000,

    learning_rate=0.05,

    depth=8,

    loss_function="RMSE",

    eval_metric="RMSE",

    random_seed=42,

    verbose=100,

    early_stopping_rounds=50
)

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\nTraining Started...\n")

start_time = time.time()

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

end_time = time.time()

training_time = end_time - start_time

print("\nTraining Completed!")

# ==========================================================
# PREDICTION
# ==========================================================

pred = model.predict(X_test)

# ==========================================================
# EVALUATION METRICS
# ==========================================================

r2 = r2_score(y_test, pred)

mae = mean_absolute_error(y_test, pred)

rmse = np.sqrt(
    mean_squared_error(y_test, pred)
)

smape_score = smape(
    y_test,
    pred
)

# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n==============================")
print(f"Training Time : {training_time:.2f} seconds")
print(f"Training Time : {training_time/60:.2f} minutes")
print(f"R² Score      : {r2:.4f}")
print(f"MAE           : {mae:.4f}")
print(f"RMSE          : {rmse:.4f}")
print(f"SMAPE         : {smape_score:.2f}%")
print("==============================")

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    "../models/CatBoost_1h.pkl"
)

print("Model Saved!")

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

prediction = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred
})

prediction.to_csv(
    "../results/CatBoost_1h_Predictions.csv",
    index=False
)

print("Predictions Saved!")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.get_feature_importance()
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "../results/CatBoost_1h_FeatureImportance.csv",
    index=False
)

print("Feature Importance Saved!")

print("\nTop 20 Important Features:")
print(importance.head(20))

print("\nDone!")