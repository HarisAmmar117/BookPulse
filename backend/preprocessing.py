"""
Applies the exact same preprocessing used in Stage 4
(`data_preprocessing_corrected.ipynb`) to a single new booking, using
the artifacts that notebook saved -- not a re-implementation that
could silently drift out of sync with it.

Pipeline, mirroring Stage 4's order:
  1. Feature engineering: total_guests, total_nights,
     previous_cancellations_ratio.
  2. Outlier treatment: cap `adr` at the training set's winsorizing
     threshold (a single new value can only ever be capped down, it
     can't be "negative" here since the schema already enforces adr >= 0).
  3. One-hot encode the categorical columns Stage 4 one-hot encoded.
  4. Frequency-encode `country` using the training-set frequency map;
     a country never seen during training maps to 0.0, correctly
     signalling "no training evidence" rather than guessing.
  5. Reindex to the exact column set/order the model was trained on --
     this single step is what correctly reproduces `drop_first`
     one-hot behaviour and silently (and correctly) zeroes out any
     rare category that Stage 4's feature selection dropped entirely.
  6. Scale the numeric columns with the already-fitted scaler
     (`transform` only -- never `fit_transform` on new data).
"""

import numpy as np
import pandas as pd

from model_loader import Artifacts
from schemas import BookingRequest


def preprocess_booking(booking: BookingRequest, artifacts: Artifacts) -> pd.DataFrame:
    row = booking.model_dump()
    df = pd.DataFrame([row])

    # --- 1. Feature engineering (same formulas as Stage 4) ---
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

    denominator = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["previous_cancellations_ratio"] = np.where(
        denominator == 0, 0, df["previous_cancellations"] / denominator
    )

    # Note: raw columns Stage 4 later dropped as redundant (e.g.
    # stays_in_week_nights/weekend_nights, distribution_channel_Direct)
    # do not need to be dropped manually here -- step 5's reindex to
    # artifacts.feature_columns removes anything the model was not
    # trained on automatically.

    # --- 2. Outlier treatment: cap adr at the Stage 4 training threshold ---
    df["adr"] = np.where(df["adr"] > artifacts.adr_upper_cap, artifacts.adr_upper_cap, df["adr"])

    # --- 3. One-hot encode the same columns Stage 4 one-hot encoded ---
    # (no drop_first here -- step 5's reindex reproduces that behaviour
    # correctly for a single row, see module docstring)
    cols_to_encode = [c for c in artifacts.low_card_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode)

    # --- 4. Frequency-encode country ---
    df["country_encoded"] = df["country"].map(artifacts.country_freq_map).fillna(0.0)
    df = df.drop(columns=["country"])

    # --- 5. Align to the exact training-time feature set/order ---
    df = df.reindex(columns=artifacts.feature_columns, fill_value=0)

    # --- 6. Scale numeric columns with the already-fitted scaler ---
    df[artifacts.numeric_cols] = artifacts.scaler.transform(df[artifacts.numeric_cols])

    return df
