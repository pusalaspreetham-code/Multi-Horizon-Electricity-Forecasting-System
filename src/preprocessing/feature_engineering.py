import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "data" / "processed"

INPUT_CSV = DATA_DIR / "Forecasting_With_Weather.csv"
OUTPUT_PARQUET = DATA_DIR / "Forecasting_Master.parquet"

METER_COL = "meter"
DATETIME_COL = "x_Timestamp"
VALUE_COL = "t_kWh"                
VOLTAGE_COL = "z_Avg Voltage (Volt)"
CURRENT_COL = "z_Avg Current (Amp)"

WEATHER_COLS = ["Temperature", "Humidity", "Precipitation", "WindSpeed", "SurfacePressure"]

CSV_READ_CHUNKSIZE = 2_000_000     
PARQUET_ROW_GROUP_SIZE = 250_000
GC_EVERY_N_METERS = 1              

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("feature_engineering")

LAG_PERIODS = [1, 3, 6]

EMA_SPANS = [3, 6, 12, 24, 48, 96, 168, 336, 720, 1440, 2880]

ROLLING_MEAN_WINDOWS = [24, 72, 168, 336, 720]
ROLLING_STD_WINDOWS = [6, 12, 24, 72, 168]
ROLLING_MAX_WINDOWS = [3, 6, 24, 72, 168]
ROLLING_MIN_WINDOWS = [3, 6, 24, 72, 168]
ROLLING_MEDIAN_WINDOWS = [6, 24, 72, 168]

DIFF_PERIODS = [6, 24, 72, 168, 336]

TREND_EMA_PAIRS = [(24, 96), (96, 336), (336, 720)]
TREND_ROLL_PAIRS = [(24, 168), (168, 720)]
HISTORICAL_ENERGY_WINDOWS = {
    "last_1h_energy": 20,
    "last_3h_energy": 60,
    "last_6h_energy": 120,
    "last_12h_energy": 240,
    "last_24h_energy": 480,
    "last_7day_energy": 3360,
    "last_14day_energy": 6720,
    "last_30day_energy": 14400,
}


# ==============================================================================
# DTYPE HELPERS
# ==============================================================================

def downcast_floats(df: pd.DataFrame, cols=None) -> None:
    cols = cols if cols is not None else df.select_dtypes(include=["float64"]).columns
    for c in cols:
        df[c] = df[c].astype(np.float32, copy=False)


def downcast_small_ints(df: pd.DataFrame, col: str, kind: str = "int16") -> None:
    if col in df.columns:
        df[col] = df[col].astype(kind, copy=False)


def add_if_missing(df: pd.DataFrame, name: str, compute_fn) -> None:
    if name not in df.columns:
        df[name] = compute_fn()



def engineer_meter(df: pd.DataFrame) -> pd.DataFrame:
    s = df[VALUE_COL].astype(np.float32)

    history = s.shift(1)

    for p in LAG_PERIODS:
        add_if_missing(df, f"lag_{p}", lambda p=p: s.shift(p))

    for span in EMA_SPANS:
        add_if_missing(df, f"ema_{span}", lambda span=span: history.ewm(span=span, adjust=False, min_periods=1).mean())

    for w in ROLLING_MEAN_WINDOWS:
        add_if_missing(df, f"rolling_mean_{w}", lambda w=w: history.rolling(window=w, min_periods=1).mean())

    for w in ROLLING_STD_WINDOWS:
        add_if_missing(df, f"rolling_std_{w}", lambda w=w: history.rolling(window=w, min_periods=1).std())

    for w in ROLLING_MAX_WINDOWS:
        add_if_missing(df, f"rolling_max_{w}", lambda w=w: history.rolling(window=w, min_periods=1).max())

    for w in ROLLING_MIN_WINDOWS:
        add_if_missing(df, f"rolling_min_{w}", lambda w=w: history.rolling(window=w, min_periods=1).min())

    for w in ROLLING_MEDIAN_WINDOWS:
        add_if_missing(df, f"rolling_median_{w}", lambda w=w: history.rolling(window=w, min_periods=1).median())

    for p in DIFF_PERIODS:
        add_if_missing(df, f"diff_{p}", lambda p=p: s.diff(p))

    for a, b in TREND_EMA_PAIRS:
        add_if_missing(df, f"trend_ema_{a}_{b}", lambda a=a, b=b: df[f"ema_{a}"] - df[f"ema_{b}"])

    for a, b in TREND_ROLL_PAIRS:
        add_if_missing(df, f"trend_roll_{a}_{b}", lambda a=a, b=b: df[f"rolling_mean_{a}"] - df[f"rolling_mean_{b}"])

    missing_energy_feats = [f for f in HISTORICAL_ENERGY_WINDOWS if f not in df.columns]
    if missing_energy_feats:
        cumsum = history.cumsum().to_numpy(dtype=np.float64)  # float64 accumulator to avoid drift
        n = len(cumsum)
        for feat_name in missing_energy_feats:
            w = HISTORICAL_ENERGY_WINDOWS[feat_name]
            if w >= n:
                shifted = np.zeros(n, dtype=np.float64)
            else:
                shifted = np.empty(n, dtype=np.float64)
                shifted[:w] = 0.0
                shifted[w:] = cumsum[:n - w]
            df[feat_name] = (cumsum - shifted).astype(np.float32)

    dt = df[DATETIME_COL]
    add_if_missing(df, "Hour", lambda: dt.dt.hour.astype(np.int8))
    add_if_missing(df, "Minute", lambda: dt.dt.minute.astype(np.int8))
    add_if_missing(df, "Month", lambda: dt.dt.month.astype(np.int8))
    add_if_missing(df, "Day", lambda: dt.dt.day.astype(np.int8))
    add_if_missing(df, "DayOfWeek", lambda: dt.dt.dayofweek.astype(np.int8))
    add_if_missing(df, "DayOfYear", lambda: dt.dt.dayofyear.astype(np.int16))
    add_if_missing(df, "WeekOfYear", lambda: dt.dt.isocalendar().week.astype(np.int8))

    hour_i16 = df["Hour"].astype(np.int16)
    minute_i16 = df["Minute"].astype(np.int16)
    add_if_missing(df, "MinutesOfDay", lambda: (hour_i16 * 60 + minute_i16).astype(np.int16))

    hour_i8 = df["Hour"]
    add_if_missing(df, "BusinessHour", lambda: hour_i8.between(9, 17).astype(np.int8))
    add_if_missing(df, "MorningPeak", lambda: hour_i8.between(6, 9).astype(np.int8))
    add_if_missing(df, "EveningPeak", lambda: hour_i8.between(18, 22).astype(np.int8))

    # ------------------------------------------------------------ CYCLICAL --
    month_col = df["Month"]
    dayofyear_col = df["DayOfYear"]

    def _hour_sin():
        hf = df["Hour"].astype(np.float32) + df["Minute"].astype(np.float32) / 60.0
        return np.sin(2 * np.pi * hf / 24.0).astype(np.float32)

    def _hour_cos():
        hf = df["Hour"].astype(np.float32) + df["Minute"].astype(np.float32) / 60.0
        return np.cos(2 * np.pi * hf / 24.0).astype(np.float32)

    add_if_missing(df, "Hour_sin", _hour_sin)
    add_if_missing(df, "Hour_cos", _hour_cos)
    add_if_missing(df, "Month_sin", lambda: np.sin(2 * np.pi * month_col.astype(np.float32) / 12.0).astype(np.float32))
    add_if_missing(df, "Month_cos", lambda: np.cos(2 * np.pi * month_col.astype(np.float32) / 12.0).astype(np.float32))
    add_if_missing(df, "DayOfYear_sin", lambda: np.sin(2 * np.pi * dayofyear_col.astype(np.float32) / 365.25).astype(np.float32))
    add_if_missing(df, "DayOfYear_cos", lambda: np.cos(2 * np.pi * dayofyear_col.astype(np.float32) / 365.25).astype(np.float32))

    for c in WEATHER_COLS:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)

    downcast_floats(df)

    return df


def build_dtype_map() -> dict:
    return {
        METER_COL: "category",
        VALUE_COL: "float32",
        VOLTAGE_COL: "float32",
        CURRENT_COL: "float32",
        **{c: "float32" for c in WEATHER_COLS},
    }


def load_and_sort(input_path: str) -> pd.DataFrame:
    log.info("Reading %s ...", input_path)
    t0 = time.time()

    dtype_map = build_dtype_map()
    df = pd.read_csv(
        input_path,
        dtype={k: v for k, v in dtype_map.items() if v != "category"},
        parse_dates=[DATETIME_COL],
    )
    if METER_COL in df.columns:
        df[METER_COL] = df[METER_COL].astype("category")

    log.info("Loaded %d rows, %d cols in %.1fs", len(df), df.shape[1], time.time() - t0)

    log.info("Sorting by %s, %s ...", METER_COL, DATETIME_COL)
    t0 = time.time()
    df.sort_values([METER_COL, DATETIME_COL], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    log.info("Sort complete in %.1fs", time.time() - t0)

    return df


def run(input_path: str = INPUT_CSV, output_path: str = OUTPUT_PARQUET) -> None:
    df = load_and_sort(input_path)

    meter_ids = df[METER_COL].cat.categories if hasattr(df[METER_COL], "cat") else df[METER_COL].unique()
    present_meters = pd.unique(df[METER_COL])
    log.info("Processing %d meters ...", len(present_meters))

    writer = None
    out_path = Path(output_path)
    if out_path.exists():
        out_path.unlink()

    grouped = df.groupby(METER_COL, sort=False, observed=True)

    n_processed = 0
    try:
        for meter_id, meter_df in tqdm(grouped, total=len(present_meters), desc="Meters", unit="meter"):
            meter_df = meter_df.reset_index(drop=True)

            featured = engineer_meter(meter_df)

            table = pa.Table.from_pandas(featured, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    str(out_path),
                    table.schema,
                    compression="snappy",
                )
            else:
                if not table.schema.equals(writer.schema):
                    table = table.cast(writer.schema)

            writer.write_table(table, row_group_size=PARQUET_ROW_GROUP_SIZE)
    
            n_processed += 1
            del meter_df, featured, table
            if n_processed % GC_EVERY_N_METERS == 0:
                gc.collect()

    finally:
        if writer is not None:
            writer.close()

    log.info("Done. Wrote %d meters to %s", n_processed, out_path)


if __name__ == "__main__":
    run()