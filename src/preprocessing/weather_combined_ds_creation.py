import pandas as pd
import os

# ==========================
# File paths
# ==========================

base_dir = os.path.dirname(os.path.abspath(__file__))

processed_path = os.path.join(base_dir, "..", "..", "data", "processed")
raw_path = os.path.join(base_dir, "..", "..", "data", "raw")

energy_path = os.path.join(processed_path, "Forecasting_Dataset.csv")
weather_path = os.path.join(raw_path, "weather_mathura_hourly.csv")
# ==========================
# Load datasets
# ==========================

energy = pd.read_csv(energy_path)
weather = pd.read_csv(weather_path)

# ==========================
# Convert timestamps
# ==========================

energy["x_Timestamp"] = pd.to_datetime(energy["x_Timestamp"])
weather["Timestamp"] = pd.to_datetime(weather["Timestamp"])

# ==========================
# Sort datasets
# ==========================

energy = energy.sort_values("x_Timestamp").reset_index(drop=True)
weather = weather.sort_values("Timestamp").reset_index(drop=True)

# ==========================
# Merge hourly weather
# ==========================

merged = pd.merge_asof(
    energy,
    weather,
    left_on="x_Timestamp",
    right_on="Timestamp",
    direction="backward"
)

# Remove duplicate timestamp column
merged.drop(columns=["Timestamp"], inplace=True)

# ==========================
# Fill missing weather values
# ==========================

weather_cols = [
    "Temperature",
    "Humidity",
    "Precipitation",
    "WindSpeed",
    "SurfacePressure"
]

merged[weather_cols] = merged[weather_cols].ffill().bfill()

# ==========================
# Save dataset
# ==========================

output_path = os.path.join(processed_path, "Forecasting_With_Weather.csv")

merged.to_csv(output_path, index=False)

print("Merged Shape:", merged.shape)

print("\nMissing Values:")
print(merged[weather_cols].isnull().sum())

print("\nSaved to:")
print(output_path)

print("\nColumns:")
print(merged.columns.tolist())