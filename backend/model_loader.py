"""
Loads the final trained model and the complete preprocessing pipeline
produced by the Stage 4 (`data_preprocessing_corrected.ipynb`) and
Stage 7 (`model_optimization.ipynb`) notebooks.

Everything is loaded once, at process startup (see `main.py`'s
lifespan handler), not per-request -- re-reading these files from disk
on every API call would be wasteful and is unnecessary since none of
them change at runtime.

Fails loudly (raises) if any required artifact is missing, rather than
starting the API in a broken state and failing confusingly on the
first real request.
"""

import json
import os
from dataclasses import dataclass
from typing import Any

import joblib


MODELS_DIR = os.environ.get("BOOKPULSE_MODELS_DIR", "../models")

REQUIRED_FILES = [
    "final_model.pkl",
    "final_model_info.json",
    "scaler.pkl",
    "country_freq_map.json",
    "feature_columns.json",
    "preprocessing_meta.json",
]


@dataclass
class Artifacts:
    model: Any
    model_info: dict
    scaler: Any
    country_freq_map: dict
    feature_columns: list
    low_card_cols: list
    numeric_cols: list
    adr_upper_cap: float


def _path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def load_artifacts() -> Artifacts:
    missing = [f for f in REQUIRED_FILES if not os.path.exists(_path(f))]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifact(s) in '{}': {}. "
            "Run the Stage 4 and Stage 7 notebooks first so these files are generated.".format(
                MODELS_DIR, ", ".join(missing)
            )
        )

    model = joblib.load(_path("final_model.pkl"))
    scaler = joblib.load(_path("scaler.pkl"))

    with open(_path("final_model_info.json")) as f:
        model_info = json.load(f)
    with open(_path("country_freq_map.json")) as f:
        country_freq_map = json.load(f)
    with open(_path("feature_columns.json")) as f:
        feature_columns = json.load(f)
    with open(_path("preprocessing_meta.json")) as f:
        meta = json.load(f)

    return Artifacts(
        model=model,
        model_info=model_info,
        scaler=scaler,
        country_freq_map=country_freq_map,
        feature_columns=feature_columns,
        low_card_cols=meta["low_card_cols"],
        numeric_cols=meta["numeric_cols"],
        adr_upper_cap=meta["adr_upper_cap"],
    )
