import gc
import os
import joblib
import numpy as np
import pandas as pd
import time
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================
def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100
print("Loading dataset...")

df = pd.read_parquet("../data/Forecasting_1m.parquet")

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
    "target_1m",
    "meter",
    "x_Timestamp",
    "z_Avg Voltage (Volt)",
    "z_Avg Current (Amp)",
    "y_Freq (Hz)"
]

feature_columns = [c for c in df.columns if c not in drop_columns]

print("Number of Features:", len(feature_columns))

# ==========================================================
# CREATE NUMPY ARRAYS (LESS MEMORY)
# ==========================================================

print("Preparing training data...")

X_train = df.loc[mask, feature_columns].to_numpy(dtype=np.float32)
y_train = df.loc[mask, "target_1m"].to_numpy(dtype=np.float32)

print("Preparing testing data...")

X_test = df.loc[~mask, feature_columns].to_numpy(dtype=np.float32)
y_test = df.loc[~mask, "target_1m"].to_numpy(dtype=np.float32)

# Free dataframe memory

gc.collect()

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ==========================================================
# XGBOOST MODEL
# ==========================================================

model = XGBRegressor(
    objective="reg:squarederror",

    n_estimators=200,
    learning_rate=0.1,

    max_depth=5,
    min_child_weight=5,

    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,

    subsample=0.8,
    colsample_bytree=0.8,

    tree_method="hist",

    random_state=42,
    early_stopping_rounds=30,

    n_jobs=-1
)

print("\nTraining Started...\n")
start_time = time.time()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)
end_time = time.time()

training_time = end_time - start_time

print(f"\nTraining Time: {training_time:.2f} seconds")
print(f"Training Time: {training_time/60:.2f} minutes")
print("\nTraining Completed!")

# ==========================================================
# PREDICTION
# ==========================================================

pred = model.predict(X_test)
test_meter = df.loc[~mask, "meter"].to_numpy()
error = np.abs(y_test - pred)

print("\n========== Error Statistics ==========")
print(pd.Series(error).describe())

print("\n========== Large Errors ==========")
print("Error > 20 :", np.sum(error > 20))
print("Error > 50 :", np.sum(error > 50))
print("Error > 100:", np.sum(error > 100))
print("Error > 200:", np.sum(error > 200))

result = pd.DataFrame({
    "Meter": test_meter,
    "Actual": y_test,
    "Predicted": pred,
    "Error": error
})
print("\n========== Top 20 Worst Predictions ==========")
print(result.sort_values("Error", ascending=False).head(20))
print("\n===== Average Error Per Meter =====")
print(result.groupby("Meter")["Error"].mean().sort_values(ascending=False))
# ==========================================================
# EVALUATION
# ==========================================================

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
smape_score = smape(
    y_test,
    pred
)
print("\n==============================")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"SMAPE    : {smape_score:.4f}")
print("==============================")

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(model, "../models/XGBoost_1m.pkl")
print("Model Saved!")

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

prediction = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred
})

prediction.to_csv(
    "../results/XGBoost_1m_Predictions.csv",
    index=False
)

print("Predictions Saved!")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "../results/XGBoost_1m_FeatureImportance.csv",
    index=False
)

print("\nTop 20 Important Features:")
print(importance.head(20))

print("\nFeature Importance Saved!")
print("\nDone!")

error = np.abs(y_test - pred)

print("Max Error :", error.max())
print("Mean Error:", error.mean())

idx = np.argmax(error)

print("Worst Actual    :", y_test[idx])
print("Worst Predicted :", pred[idx])
print("Worst Error     :", error[idx])

print(y_test.min(), y_test.max())
print(pred.min(), pred.max())