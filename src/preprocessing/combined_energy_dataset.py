import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# =====================================================
# 1. Locate raw folder
# =====================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(base_dir, "..", "..", "data", "raw")

# =====================================================
# 2. Read all CSV files automatically
# =====================================================

csv_files = [f for f in os.listdir(raw_path) if f.endswith(".csv")]

print("CSV Files Found:")
for file in csv_files:
    print(file)

dfs = []

for file in csv_files:
    print(f"Reading {file}...")
    temp = pd.read_csv(os.path.join(raw_path, file))
    dfs.append(temp)

# =====================================================
# 3. Combine all datasets
# =====================================================

df = pd.concat(dfs, ignore_index=True)

print("\nTotal Rows:", len(df))

# =====================================================
# 4. Remove duplicate rows
# =====================================================

duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

df.drop_duplicates(inplace=True)

# =====================================================
# 5. Convert Timestamp
# =====================================================

df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"])

# =====================================================
# 6. Sort by Timestamp
# =====================================================

df.sort_values(["meter", "x_Timestamp"], inplace=True)

# =====================================================
# 7. Time Feature Engineering
# =====================================================

df["Year"] = df["x_Timestamp"].dt.year
df["Month"] = df["x_Timestamp"].dt.month
df["Day"] = df["x_Timestamp"].dt.day

df["Hour"] = df["x_Timestamp"].dt.hour
df["Minute"] = df["x_Timestamp"].dt.minute

df["DayOfWeek"] = df["x_Timestamp"].dt.dayofweek

df["Weekend"] = (df["DayOfWeek"] >= 5).astype(int)

df["Quarter"] = df["x_Timestamp"].dt.quarter

df["WeekOfYear"] = df["x_Timestamp"].dt.isocalendar().week.astype(int)

df["DayOfYear"] = df["x_Timestamp"].dt.dayofyear

# =====================================================
# 8. Season Feature
# =====================================================

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8]:
        return "Monsoon"
    else:
        return "Autumn"

df["Season"] = df["Month"].apply(get_season)

# =====================================================
# 9. Encode Meter
# =====================================================

meter_encoder = LabelEncoder()
df["meter"] = meter_encoder.fit_transform(df["meter"])

# =====================================================
# 10. Encode Season
# =====================================================

season_encoder = LabelEncoder()
df["Season"] = season_encoder.fit_transform(df["Season"])

# =====================================================
# 11. Missing Values
# =====================================================

print("\nMissing Values:")
print(df.isnull().sum())

df.dropna(inplace=True)

# =====================================================
# 12. Reset Index
# =====================================================

df.reset_index(drop=True, inplace=True)

# =====================================================
# 13. Dataset Information
# =====================================================

print("\nFinal Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst Five Rows:")
print(df.head())

# =====================================================
# 14. Save Dataset
# =====================================================
processed_path = os.path.join(base_dir, "..", "..", "data", "processed")
os.makedirs(processed_path, exist_ok=True)
output_file = os.path.join(processed_path, "Combined_Energy_Dataset.csv")
df.to_csv(output_file, index=False)

print("\nDataset saved successfully!")
print(output_file)