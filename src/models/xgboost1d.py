import gc
import os
import joblib
import numpy as np
import pandas as pd

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

df = pd.read_parquet("../data/Forecasting_1d.parquet")

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
    "target_1d",
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
y_train = df.loc[mask, "target_1d"].to_numpy(dtype=np.float32)

print("Preparing testing data...")

X_test = df.loc[~mask, feature_columns].to_numpy(dtype=np.float32)
y_test = df.loc[~mask, "target_1d"].to_numpy(dtype=np.float32)

# Free dataframe memory
del df
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

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

print("\nTraining Completed!")

# ==========================================================
# PREDICTION
# ==========================================================

pred = model.predict(X_test)

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

joblib.dump(model, "../models/XGBoost_1d.pkl")
print("Model Saved!")

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

prediction = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred
})

prediction.to_csv(
    "../results/XGBoost_1d_Predictions.csv",
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
    "../results/XGBoost_1d_FeatureImportance.csv",
    index=False
)

print("\nTop 20 Important Features:")
print(importance.head(20))

print("\nFeature Importance Saved!")
print("\nDone!")