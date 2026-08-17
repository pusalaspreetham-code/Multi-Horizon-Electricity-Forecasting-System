import os
import pandas as pd

# =====================================================
# Paths
# =====================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
processed_path = os.path.join(base_dir, "..", "..", "data", "processed")

input_file = os.path.join(processed_path, "Combined_Energy_Dataset.csv")
output_file = os.path.join(processed_path, "Forecasting_Dataset.csv")

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(input_file)

print("Original Shape :", df.shape)

# =====================================================
# Timestamp
# =====================================================

df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"])

# =====================================================
# Sort by Meter and Time
# =====================================================

df = df.sort_values(["meter", "x_Timestamp"]).reset_index(drop=True)

# =====================================================
# Lag Features
# =====================================================

print("Creating lag features...")

df["lag_1"] = df.groupby("meter")["t_kWh"].shift(1)
df["lag_2"] = df.groupby("meter")["t_kWh"].shift(2)
df["lag_3"] = df.groupby("meter")["t_kWh"].shift(3)

df["lag_6"] = df.groupby("meter")["t_kWh"].shift(6)

df["lag_12"] = df.groupby("meter")["t_kWh"].shift(12)

df["lag_24"] = df.groupby("meter")["t_kWh"].shift(24)

# =====================================================
# Rolling Mean
# =====================================================

print("Creating rolling mean...")

df["rolling_mean_3"] = (
    df.groupby("meter")["t_kWh"]
      .transform(lambda x: x.shift(1).rolling(3).mean())
)

df["rolling_mean_6"] = (
    df.groupby("meter")["t_kWh"]
      .transform(lambda x: x.shift(1).rolling(6).mean())
)

df["rolling_mean_12"] = (
    df.groupby("meter")["t_kWh"]
      .transform(lambda x: x.shift(1).rolling(12).mean())
)

# =====================================================
# Rolling Std
# =====================================================

print("Creating rolling std...")

df["rolling_std_6"] = (
    df.groupby("meter")["t_kWh"]
      .transform(lambda x: x.shift(1).rolling(6).std())
)

df["rolling_std_12"] = (
    df.groupby("meter")["t_kWh"]
      .transform(lambda x: x.shift(1).rolling(12).std())
)

# =====================================================
# Drop NaN
# =====================================================

print("Removing initial rows with NaN...")

df = df.dropna().reset_index(drop=True)

print("New Shape :", df.shape)

# =====================================================
# Save
# =====================================================

df.to_csv(output_file, index=False)

print("\nForecasting dataset created successfully!")
print(output_file)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())