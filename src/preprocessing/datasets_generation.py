

import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "data" / "processed"

INPUT_PARQUET = DATA_DIR / "Forecasting_Master.parquet"
OUTPUT_DIR = DATA_DIR

METER_COL = "meter"
DATETIME_COL = "x_Timestamp"
VALUE_COL = "t_kWh"

OUTPUT_3M = OUTPUT_DIR / "Forecasting_3m.parquet"
OUTPUT_1H = OUTPUT_DIR / "Forecasting_1h.parquet"
OUTPUT_1D = OUTPUT_DIR / "Forecasting_1d.parquet"
OUTPUT_1M = OUTPUT_DIR / "Forecasting_1m.parquet"

PARQUET_COMPRESSION = "snappy"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("generate_targets")


# ==============================================================================
# LOAD
# ==============================================================================

def load_master(path: Path) -> pd.DataFrame:
    log.info("Reading %s ...", path)
    t0 = time.time()
    df = pd.read_parquet(path, engine="pyarrow")
    log.info("Loaded %d rows, %d cols in %.1fs", len(df), df.shape[1], time.time() - t0)

    if not np.issubdtype(df[DATETIME_COL].dtype, np.datetime64):
        log.info("Parsing %s as datetime ...", DATETIME_COL)
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])

    # Data is documented as already sorted by [meter, x_Timestamp]; groupby
    # correctness below does not actually require sortedness (groupby keys
    # define the grouping regardless of row order), but cumsum() ordering
    # WITHIN each group must follow chronological order, so we defensively
    # verify rather than blindly trust it.
    is_sorted = df[[METER_COL, DATETIME_COL]].equals(
        df[[METER_COL, DATETIME_COL]].sort_values([METER_COL, DATETIME_COL], kind="mergesort")
    )
    if not is_sorted:
        log.warning("Input not sorted by [%s, %s] -- sorting now.", METER_COL, DATETIME_COL)
        df.sort_values([METER_COL, DATETIME_COL], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
    else:
        log.info("Confirmed input is sorted by [%s, %s].", METER_COL, DATETIME_COL)

    return df


# ==============================================================================
# TARGET COMPUTATION (vectorized, no row/meter-level Python loops)
# ==============================================================================

def compute_target_3m(df: pd.DataFrame) -> pd.Series:
    """Next row's t_kWh, per meter. NaN for the last row of each meter."""
    return (
        df.groupby(METER_COL, sort=False, observed=True)[VALUE_COL]
        .shift(-1)
        .astype(np.float32)
    )


def compute_remaining_until_period_end(df: pd.DataFrame, period_key: pd.Series) -> pd.Series:
    """
    Remaining sum of t_kWh strictly AFTER the current row, within the same
    (meter, period_key) bucket. Single groupby pass, fully vectorized.
    """
    grp = df.groupby([df[METER_COL], period_key], sort=False, observed=True)[VALUE_COL]

    cumsum_inclusive = grp.cumsum().to_numpy(dtype=np.float64)
    total = grp.transform("sum").to_numpy(dtype=np.float64)

    target = (total - cumsum_inclusive).astype(np.float32)
    return pd.Series(target, index=df.index)


def hour_bucket(df: pd.DataFrame) -> pd.Series:
    return df[DATETIME_COL].dt.floor("h")


def day_bucket(df: pd.DataFrame) -> pd.Series:
    return df[DATETIME_COL].dt.floor("D")


def month_bucket(df: pd.DataFrame) -> pd.Series:
    # Integer-encoded year*100+month is faster to hash/group than Period
    # objects while still uniquely resetting every calendar month.
    dt = df[DATETIME_COL]
    return (dt.dt.year.astype(np.int32) * 100 + dt.dt.month.astype(np.int32))


# ==============================================================================
# VALIDATION / REPORTING
# ==============================================================================

def validate_and_report(name: str, df: pd.DataFrame, target_col: str) -> None:
    n_nan = int(df[target_col].isna().sum())
    log.info("=" * 60)
    log.info("Dataset name       : %s", name)
    log.info("Number of rows     : %d", len(df))
    log.info("Number of columns  : %d", df.shape[1])
    log.info("Target mean        : %.6f", df[target_col].mean())
    log.info("Target min         : %.6f", df[target_col].min())
    log.info("Target max         : %.6f", df[target_col].max())
    log.info("Number of NaN vals : %d", n_nan)
    log.info("=" * 60)


# ==============================================================================
# BUILD + SAVE ONE TARGET DATASET
# ==============================================================================

def build_and_save(
    df: pd.DataFrame,
    target_col: str,
    target_values: pd.Series,
    output_path: Path,
    dataset_name: str,
) -> None:
    log.info("Building %s ...", dataset_name)
    t0 = time.time()

    df[target_col] = target_values
    n_before = len(df)
    out = df.dropna(subset=[target_col])
    n_dropped = n_before - len(out)
    log.info("Dropped %d rows with NaN %s", n_dropped, target_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
        compression=PARQUET_COMPRESSION,
    )
    log.info("Wrote %s (%d rows, %d cols) in %.1fs", output_path, len(out), out.shape[1], time.time() - t0)

    validate_and_report(dataset_name, out, target_col)

    del out
    df.drop(columns=[target_col], inplace=True)
    gc.collect()


# ==============================================================================
# MAIN
# ==============================================================================

def run() -> None:
    df = load_master(INPUT_PARQUET)

    # ---- target_3m ----------------------------------------------------------
    target_3m = compute_target_3m(df)
    build_and_save(df, "target_3m", target_3m, OUTPUT_3M, "Forecasting_3m")
    del target_3m
    gc.collect()

    # ---- target_1h ------------------------------------------------------------
    target_1h = compute_remaining_until_period_end(df, hour_bucket(df))
    build_and_save(df, "target_1h", target_1h, OUTPUT_1H, "Forecasting_1h")
    del target_1h
    gc.collect()

    # ---- target_1d ------------------------------------------------------------
    target_1d = compute_remaining_until_period_end(df, day_bucket(df))
    build_and_save(df, "target_1d", target_1d, OUTPUT_1D, "Forecasting_1d")
    del target_1d
    gc.collect()

    # ---- target_1m ------------------------------------------------------------
    target_1m = compute_remaining_until_period_end(df, month_bucket(df))
    build_and_save(df, "target_1m", target_1m, OUTPUT_1M, "Forecasting_1m")
    del target_1m
    gc.collect()

    log.info("All four target datasets generated successfully.")


if __name__ == "__main__":
    run()