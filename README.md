# BookPulse API — Hotel Booking Cancellation Prediction (Stage 9)

A FastAPI backend that loads the final trained model from Stage 7 and
the full preprocessing pipeline from Stage 4, and serves cancellation
predictions for a single new booking.

## Expected folder layout

This backend expects to sit alongside the `models/` folder your
Stage 4 and Stage 7 notebooks already write to:

```
project-root/
├── dataset/
├── models/
│   ├── final_model.pkl
│   ├── final_model_info.json
│   ├── scaler.pkl
│   ├── country_freq_map.json
│   ├── feature_columns.json
│   └── preprocessing_meta.json
├── requirements.txt
└── backend/
    ├── main.py
    ├── schemas.py
    ├── preprocessing.py
    └── model_loader.py
```

If you keep a different layout, set the `BOOKPULSE_MODELS_DIR`
environment variable to point at your `models/` folder instead of
editing the code.

## Setup

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Run the API

Start the API from the `backend/` folder:

```bash
cd backend
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger
docs — you can fill in the example booking and call `/predict`
directly from the browser, no separate frontend needed to test it.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check; confirms the model is loaded |
| `GET` | `/model-info` | Returns the final model's name and Stage 7 test metrics |
| `POST` | `/predict` | Accepts one booking, returns a cancellation prediction |

## Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hotel": "City Hotel",
    "lead_time": 85,
    "arrival_date_year": 2026,
    "arrival_date_month": "July",
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 2,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 3,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "adr": 110.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1
  }'
```

Example response:

```json
{
  "prediction": "Not Cancelled",
  "will_cancel": false,
  "cancellation_probability": 0.44,
  "risk_level": "Medium",
  "estimated_revenue_at_risk": 242.0,
  "model_used": "Random Forest (tuned)"
}
```

## Input validation

- Every field is type- and range-checked by Pydantic (e.g. `adr >= 0`,
  `adults <= 10`). Categorical fields (`hotel`, `meal`,
  `market_segment`, `deposit_type`, etc.) only accept the exact
  category values seen in the Hotel Booking Demand dataset — an
  invalid one returns a `422` listing the allowed values.
- A booking with zero total guests, or zero total nights, is rejected
  with a clear `422` message — this mirrors the exact data-quality
  rule established during Stage 3/4 EDA (such rows were treated as
  invalid and removed from the training data itself).
- A `country` not seen during training, or a rare room-type/category
  that Stage 4's feature selection dropped, is **not** rejected — it
  is handled gracefully by the preprocessing pipeline (treated as "no
  signal for this category"), since real-world bookings should not
  fail outright just because they're slightly unusual.

## How the preprocessing pipeline stays correct

`preprocessing.py` does **not** hardcode which one-hot columns or
categories exist. Instead, it reindexes every new booking against the
literal `feature_columns.json` your Stage 4 notebook produced. This
means:

- If you rerun Stage 4 with different feature-selection decisions,
  the backend automatically adapts — no code changes needed here.
- Categories that Stage 4 dropped (e.g. rare room types, `meal_FB`)
  are automatically zeroed out for new bookings too, exactly
  reproducing `pd.get_dummies(..., drop_first=True)` behaviour.
