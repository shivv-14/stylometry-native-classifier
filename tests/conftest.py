from pathlib import Path

import pandas as pd
import pytest

from stylometry.config import load_config

FIXTURE = Path(__file__).parent / "fixtures" / "SYNTHETIC_samples.csv"


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    # keep tests fast: small grids, no analyzer tuning, few folds
    c["models"]["lr"]["C"] = [0.1, 1]
    c["models"]["svm"]["C"] = [0.1, 1]
    c["models"]["svm"]["calibration_cv"] = 2
    c["models"]["tune_char_analyzer"] = []
    c["split"]["cv_folds"] = 3
    c["features"]["char_ngrams"]["min_df"] = 1
    c["evaluation"]["bootstrap_samples"] = 50
    return c


@pytest.fixture(scope="session")
def synthetic_df():
    """SYNTHETIC fixture: template texts for unit tests only, never for results."""
    return pd.read_csv(FIXTURE)
