from pathlib import Path
import pandas as pd
from .config import DATA_PATH, REQUIRED_COLUMNS, TARGET

def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")
    bad = set(df[TARGET].dropna().unique()) - {0, 1}
    if bad:
        raise ValueError(f"{TARGET} must contain only 0 and 1; found {sorted(bad)}.")
