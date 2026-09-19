# ⚡ Multi-Horizon Electricity Consumption Forecasting

A machine learning pipeline for forecasting electricity consumption across **1-hour, 1-day, and 1-month horizons** using historical smart-meter data, engineered temporal features, and weather information.

The project compares **XGBoost, LightGBM, and CatBoost** models and generates predictions and feature-importance results for each forecasting horizon.

---

## 📌 Overview

Electricity consumption depends on historical usage patterns, time-related behavior, and environmental conditions.

This project develops an end-to-end forecasting pipeline that:

* Processes historical smart-meter electricity data.
* Integrates weather information using timestamps.
* Engineers temporal, statistical, lag, rolling, EMA, trend, and historical-energy features.
* Creates independent forecasting datasets for three horizons.
* Trains XGBoost, LightGBM, and CatBoost models.
* Evaluates and stores model predictions.
* Generates feature-importance results for model interpretation.

### Forecasting Horizons

| Horizon     | Prediction Target                                                    |
| ----------- | -------------------------------------------------------------------- |
| **1 Hour**  | Remaining electricity consumption until the end of the current hour  |
| **1 Day**   | Remaining electricity consumption until the end of the current day   |
| **1 Month** | Remaining electricity consumption until the end of the current month |

These represent **three separate forecasting tasks**, rather than three random splits of the original dataset.

---

## 🎯 Objectives

The main objectives of the project are to:

* Forecast electricity consumption at multiple time horizons.
* Capture historical consumption patterns using lag and rolling features.
* Incorporate weather conditions into the forecasting pipeline.
* Compare three gradient-boosting algorithms.
* Analyze the features that contribute most to model predictions.
* Build a reproducible end-to-end machine learning workflow.

---

## 🔄 Project Pipeline

```text
                Raw Smart-Meter Data
                         │
                         ▼
              Data Cleaning & Processing
                         │
                         ▼
                Dataset Combination
                         │
                         ▼
                 Weather Integration
                         │
                         ▼
                  Feature Engineering
                         │
                         ▼
              Forecast Target Creation
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          1-Hour       1-Day      1-Month
          Dataset      Dataset      Dataset
             │           │           │
             └───────────┼───────────┘
                         ▼
                  Model Training
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       XGBoost        LightGBM       CatBoost
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Predictions
                         │
                         ▼
             Feature Importance Analysis
```

---

## 📊 Dataset

The project combines **historical smart-meter electricity consumption data** with weather information.

### Electricity Data

| Feature Category        | Examples                    |
| ----------------------- | --------------------------- |
| Consumption             | `t_kWh`                     |
| Electrical Measurements | Voltage, Current, Frequency |
| Identification          | Meter ID                    |
| Temporal                | Timestamp                   |

### Weather Data

| Feature         | Description                                |
| --------------- | ------------------------------------------ |
| Temperature     | Temperature at the corresponding timestamp |
| Humidity        | Atmospheric humidity                       |
| Precipitation   | Precipitation amount                       |
| WindSpeed       | Wind speed                                 |
| SurfacePressure | Surface pressure                           |

Weather variables are matched with electricity records using timestamps.

---

## 🧠 Feature Engineering

Feature engineering is one of the main components of the forecasting pipeline.

### Feature Categories

| Category               | Examples                            | Purpose                                 |
| ---------------------- | ----------------------------------- | --------------------------------------- |
| **Time Features**      | Hour, Month, DayOfWeek, Season      | Capture recurring time patterns         |
| **Cyclical Features**  | `Hour_sin`, `Hour_cos`              | Represent periodic time behavior        |
| **Lag Features**       | `lag_1`, `lag_6`, `lag_24`          | Capture previous consumption            |
| **Rolling Statistics** | Mean, Std, Max, Min, Median         | Capture recent consumption behavior     |
| **EMA Features**       | `ema_3`, `ema_24`, `ema_168`        | Give more weight to recent observations |
| **Trend Features**     | `trend_ema_24_96`                   | Capture increasing/decreasing trends    |
| **Historical Energy**  | `last_1h_energy`, `last_24h_energy` | Represent historical energy consumption |
| **Weather Features**   | Temperature, Humidity, WindSpeed    | Capture environmental effects           |

### Historical Energy Windows

The pipeline calculates historical consumption over multiple time windows:

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

This allows the models to capture both short-term and long-term consumption patterns.

---

## 🎯 Forecasting Targets

Three independent forecasting datasets are generated:

| Dataset                  | Target                                                   |
| ------------------------ | -------------------------------------------------------- |
| `Forecasting_1h.parquet` | Remaining consumption until the end of the current hour  |
| `Forecasting_1d.parquet` | Remaining consumption until the end of the current day   |
| `Forecasting_1m.parquet` | Remaining consumption until the end of the current month |

Each horizon is treated as a separate machine learning problem.

---

## 🤖 Machine Learning Models

Three gradient-boosting models are trained independently for each forecasting horizon.

| Model        | 1 Hour | 1 Day | 1 Month |
| ------------ | :----: | :---: | :-----: |
| **XGBoost**  |    ✓   |   ✓   |    ✓    |
| **LightGBM** |    ✓   |   ✓   |    ✓    |
| **CatBoost** |    ✓   |   ✓   |    ✓    |

### XGBoost

XGBoost is used to model nonlinear relationships between electricity consumption and the engineered features.

### LightGBM

LightGBM provides an efficient gradient-boosting implementation suitable for large tabular datasets.

### CatBoost

CatBoost provides an additional gradient-boosting approach for comparison with XGBoost and LightGBM.

The models are trained **independently**. Their predictions are not combined into an ensemble.

---

## 📈 Results

The pipeline generates predictions and feature-importance results for every model and forecasting horizon.

### Output Categories

| Output             | Location                      |
| ------------------ | ----------------------------- |
| Model Predictions  | `results/predictions/`        |
| Feature Importance | `results/feature_importance/` |

### Generated Predictions

```text
results/predictions/

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

### Feature Importance

```text
results/feature_importance/

├── CatBoost_1h_FeatureImportance.csv
├── CatBoost_1d_FeatureImportance.csv
├── CatBoost_1m_FeatureImportance.csv
├── LightGBM_1h_FeatureImportance.csv
├── LightGBM_1d_FeatureImportance.csv
├── LightGBM_1m_FeatureImportance.csv
├── XGBoost_1h_FeatureImportance.csv
├── XGBoost_1d_FeatureImportance.csv
└── XGBoost_1m_FeatureImportance.csv
```

---

## 🗂️ Project Structure

```text
Multi-Horizon-Electricity-Forecasting-System/
│
├── README.md
├── requirements.txt
├── .gitignore
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
    └── predictions/
```

---

## 🛠️ Technologies

| Technology       | Purpose                          |
| ---------------- | -------------------------------- |
| **Python**       | Main programming language        |
| **Pandas**       | Data processing and manipulation |
| **NumPy**        | Numerical operations             |
| **Scikit-learn** | ML utilities and preprocessing   |
| **XGBoost**      | Gradient-boosting model          |
| **LightGBM**     | Gradient-boosting model          |
| **CatBoost**     | Gradient-boosting model          |
| **PyArrow**      | Parquet data processing          |
| **Joblib**       | Model/data serialization         |
| **Matplotlib**   | Visualization                    |
| **Requests**     | Weather API requests             |
| **tqdm**         | Progress tracking                |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/pusalaspreetham-code/Multi-Horizon-Electricity-Forecasting-System.git

cd Multi-Horizon-Electricity-Forecasting-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the pipeline in the following order:

### Step 1 — Process Raw Data

Process and clean the raw smart-meter datasets.

### Step 2 — Combine Electricity Data

Combine the required electricity datasets into a unified dataset.

### Step 3 — Integrate Weather Data

Retrieve and integrate weather information using timestamps.

### Step 4 — Feature Engineering

Generate:

* Time features
* Cyclical features
* Lag features
* Rolling statistics
* EMA features
* Trend features
* Historical energy features
* Weather features

### Step 5 — Generate Forecasting Datasets

Create independent datasets for:

```text
1 Hour
1 Day
1 Month
```

### Step 6 — Train Models

Train:

```text
XGBoost
LightGBM
CatBoost
```

for each forecasting horizon.

### Step 7 — Generate Predictions

Store model predictions in:

```text
results/predictions/
```

### Step 8 — Analyze Feature Importance

Store feature-importance results in:

```text
results/feature_importance/
```

---

## 🔮 Future Improvements

Potential extensions to the project include:

* Hyperparameter optimization
* Additional feature-selection techniques
* Evaluation on more unseen meters
* Deep learning models such as LSTM
* Real-time electricity forecasting
* Additional weather variables
* Additional forecasting horizons
* Automated model evaluation and comparison

---

## 📌 Conclusion

This project implements an end-to-end machine learning pipeline for **multi-horizon electricity consumption forecasting**.

The system combines historical smart-meter consumption, temporal patterns, lag and rolling statistics, exponential moving averages, trend features, historical energy windows, and weather information.

Three independent gradient-boosting models — **XGBoost, LightGBM, and CatBoost** — are trained across **1-hour, 1-day, and 1-month forecasting horizons**.

The resulting predictions and feature-importance outputs provide a basis for comparing model behavior across different forecasting tasks.
