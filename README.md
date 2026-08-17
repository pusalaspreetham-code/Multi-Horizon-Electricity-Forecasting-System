# Electricity Consumption Forecasting Using Machine Learning

## Overview

This project focuses on forecasting electricity consumption using historical smart-meter data and weather information. The workflow covers data preprocessing, weather integration, feature engineering, forecasting-target creation, machine learning model training, prediction, and feature-importance analysis.

Three forecasting horizons are considered:

| Forecasting Horizon | Prediction |
|---|---|
| 1 Hour | Remaining electricity consumption until the end of the current hour |
| 1 Day | Remaining electricity consumption until the end of the current day |
| 1 Month | Remaining electricity consumption until the end of the current month |

Three machine learning models are trained independently:

- XGBoost
- LightGBM
- CatBoost


## Objective

The main objectives of this project are to:

- Forecast electricity consumption at different time horizons.
- Use historical electricity consumption patterns as forecasting features.
- Incorporate weather information into the prediction process.
- Compare XGBoost, LightGBM, and CatBoost.
- Identify the most important features contributing to predictions.

---

## Project Workflow

```text
Raw Smart-Meter Data
        ↓
Data Cleaning & Preprocessing
        ↓
Dataset Combination
        ↓
Weather Integration
        ↓
Feature Engineering
        ↓
Forecasting Target Creation
        ↓
1-Hour / 1-Day / 1-Month Datasets
        ↓
Model Training
        ↓
Prediction
        ↓
Feature Importance Analysis
```

---

## Dataset

The project uses smart-meter electricity consumption data combined with weather information.

### Electricity Features

| Category | Examples |
|---|---|
| Consumption | `t_kWh` |
| Electrical measurements | Voltage, Current, Frequency |
| Identification | Meter ID |
| Time | Timestamp |

### Weather Features

| Feature | Description |
|---|---|
| Temperature | Temperature at the corresponding time |
| Humidity | Atmospheric humidity |
| Precipitation | Precipitation amount |
| WindSpeed | Wind speed |
| SurfacePressure | Surface pressure |

Weather information is integrated with the electricity data using timestamps.

---

## Feature Engineering

Several groups of features are generated from the historical electricity data and timestamps.

| Feature Group | Examples | Purpose |
|---|---|---|
| Time features | Hour, Month, DayOfWeek, Season | Capture time-based consumption patterns |
| Cyclical features | Hour_sin, Hour_cos | Represent repeating time cycles |
| Lag features | lag_1, lag_6, lag_24 | Capture previous consumption |
| Rolling statistics | Rolling Mean, Std, Max, Min, Median | Capture recent consumption behaviour |
| EMA | ema_3, ema_24, ema_168 | Give greater importance to recent observations |
| Trend features | trend_ema_24_96 | Capture increasing or decreasing trends |
| Historical energy | last_1h_energy, last_24h_energy | Represent previous energy consumption |
| Weather | Temperature, Humidity, WindSpeed | Capture weather-related effects |

### Historical Energy Features

The following historical energy windows are used:

```text
last_1h_energy
last_3h_energy
last_6h_energy
last_12h_energy
last_24h_energy
last_7day_energy
last_14day_energy
last_30day_energy
```

---

## Forecasting Targets

The project creates three different forecasting tasks from the processed data.

| Dataset | Target |
|---|---|
| `Forecasting_1h.parquet` | Remaining consumption until the end of the current hour |
| `Forecasting_1d.parquet` | Remaining consumption until the end of the current day |
| `Forecasting_1m.parquet` | Remaining consumption until the end of the current month |

These are **three different forecasting tasks**, not three random portions of the original dataset.

---

## Machine Learning Models

| Model | 1 Hour | 1 Day | 1 Month |
|---|:---:|:---:|:---:|
| XGBoost | ✓ | ✓ | ✓ |
| LightGBM | ✓ | ✓ | ✓ |
| CatBoost | ✓ | ✓ | ✓ |

### XGBoost

XGBoost is a gradient boosting algorithm based on decision trees. It is used to learn nonlinear relationships between electricity consumption and the engineered features.

### LightGBM

LightGBM is a gradient boosting framework designed for efficient training on large datasets.

### CatBoost

CatBoost is a gradient boosting algorithm based on decision trees and is used as an additional model for comparison.

The three models are trained independently. Their predictions are not combined in the final experiments.

---

## Results

The project generates two main types of results.

| Result | Location |
|---|---|
| Model predictions | `results/predictions/` |
| Feature importance | `results/feature_importance/` |

Predictions and feature-importance results are generated for all three models and all three forecasting horizons.

---

## Project Structure

```text
electricity-consumption-forecasting/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── CEEW - Smart meter data Mathura 2019.csv
│   │   ├── CEEW - Smart meter data Mathura 2020.csv
│   │   └── SM Cleaned Data MH2021.csv
│   │
│   └── processed/
│       ├── Combined_Energy_Dataset.csv
│       ├── Forecasting_Dataset.csv
│       ├── Forecasting_With_Weather.csv
│       ├── Forecasting_Master.parquet
│       ├── Forecasting_1h.parquet
│       ├── Forecasting_1d.parquet
│       └── Forecasting_1m.parquet
│
├── src/
│   ├── preprocessing/
│   │   ├── combined_energy_dataset.py
│   │   ├── datasets_generation.py
│   │   ├── forecasting_dataset_creation.py
│   │   ├── feature_engineering.py
│   │   └── weather_combined_ds_creation.py
│   │
│   └── models/
│       ├── catboost_1h.py
│       ├── catboost_1d.py
│       ├── catboost_1m.py
│       ├── lightgbm_1h.py
│       ├── lightgbm_1d.py
│       ├── lightgbm_1m.py
│       ├── xgboost_1h.py
│       ├── xgboost_1d.py
│       └── xgboost_1m.py
│
└── results/
    ├── feature_importance/
    │   ├── CatBoost_1h_FeatureImportance.csv
    │   ├── CatBoost_1d_FeatureImportance.csv
    │   ├── CatBoost_1m_FeatureImportance.csv
    │   ├── LightGBM_1h_FeatureImportance.csv
    │   ├── LightGBM_1d_FeatureImportance.csv
    │   ├── LightGBM_1m_FeatureImportance.csv
    │   ├── XGBoost_1h_FeatureImportance.csv
    │   ├── XGBoost_1d_FeatureImportance.csv
    │   └── XGBoost_1m_FeatureImportance.csv
    │
    └── predictions/
        ├── CatBoost_1h_Predictions.csv
        ├── CatBoost_1d_Predictions.csv
        ├── CatBoost_1m_Predictions.csv
        ├── LightGBM_1h_Predictions.csv
        ├── LightGBM_1d_Predictions.csv
        ├── LightGBM_1m_Predictions.csv
        ├── XGBoost_1h_Predictions.csv
        ├── XGBoost_1d_Predictions.csv
        └── XGBoost_1m_Predictions.csv
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning utilities |
| XGBoost | Forecasting model |
| LightGBM | Forecasting model |
| CatBoost | Forecasting model |
| PyArrow | Parquet data processing |
| Joblib | Model/data serialization |
| Matplotlib | Visualization |
| Requests | API requests |
| tqdm | Progress tracking |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/pusalaspreetham-code/Multi-Horizon-Electricity-Forecasting-System.git
cd electricity-consumption-forecasting
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the project in the following order:

1. Process the raw electricity datasets.
2. Combine the required electricity datasets.
3. Integrate weather information.
4. Perform feature engineering.
5. Generate the forecasting datasets.
6. Train the XGBoost, LightGBM, and CatBoost models.
7. Generate predictions.
8. Generate feature-importance results.

The generated outputs are stored in the `results/` directory.

---

## Future Improvements

Possible future improvements include:

- Systematic hyperparameter tuning
- Additional feature-selection methods
- Testing with more unseen meters
- Deep learning models such as LSTM
- Real-time electricity forecasting
- Additional weather variables
- Additional forecasting horizons

---

## Conclusion

This project develops a machine learning pipeline for electricity consumption forecasting across 1-hour, 1-day, and 1-month horizons.

Historical electricity consumption, time-based features, rolling statistics, EMA, trend features, historical energy features, and weather information are used to provide the models with relevant information for forecasting.

XGBoost, LightGBM, and CatBoost are trained independently and their predictions and feature-importance results are stored for comparison and analysis.#   M u l t i - H o r i z o n - E l e c t r i c i t y - F o r e c a s t i n g - S y s t e m  
 