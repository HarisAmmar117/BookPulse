"""
BookPulse -- Hotel Booking Cancellation Prediction API (Stage 9)

Run from inside the `backend/` folder:

    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.

Expects the sibling folder `../models/` to contain the artifacts saved
by the Stage 4 and Stage 7 notebooks:
  final_model.pkl, final_model_info.json, scaler.pkl,
  country_freq_map.json, feature_columns.json, preprocessing_meta.json
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_loader import Artifacts, load_artifacts
from preprocessing import preprocess_booking
from schemas import (
    BookingRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bookpulse")

# Populated at startup by the lifespan handler below; left as None until
# then so a request arriving before startup finishes fails clearly
# instead of crashing on a missing global.
artifacts: Artifacts | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifacts
    logger.info("Loading model and preprocessing artifacts...")
    artifacts = load_artifacts()
    logger.info("Loaded model: %s", artifacts.model_info.get("model_name", "unknown"))
    yield
    artifacts = None


app = FastAPI(
    title="BookPulse -- Hotel Booking Cancellation Prediction API",
    description="Predicts whether a hotel booking is likely to be cancelled, "
                 "for the IT3051 Fundamentals of Data Mining mini project.",
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS for local frontend development (Stage 10). Tighten
# `allow_origins` to the real frontend's origin before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Reshapes FastAPI's default 422 body into a flatter, friendlier
    format for the frontend to display directly, field by field."""
    errors = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": "Invalid booking input.", "errors": errors})


def _get_artifacts() -> Artifacts:
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try again shortly.")
    return artifacts


@app.get("/", include_in_schema=False)
def root():
    return {"message": "BookPulse API is running. See /docs for usage."}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=artifacts is not None)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    a = _get_artifacts()
    return ModelInfoResponse(**a.model_info)


@app.post("/predict", response_model=PredictionResponse)
def predict(booking: BookingRequest):
    """
    Accepts a single new booking, applies the exact Stage 4
    preprocessing pipeline to it, and returns the final model's
    cancellation prediction.
    """
    a = _get_artifacts()

    try:
        features = preprocess_booking(booking, a)
        probability = float(a.model.predict_proba(features)[0, 1])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    will_cancel = probability >= 0.5
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    total_nights = booking.stays_in_weekend_nights + booking.stays_in_week_nights
    estimated_revenue_at_risk = round(probability * booking.adr * total_nights, 2)

    return PredictionResponse(
        prediction="Cancelled" if will_cancel else "Not Cancelled",
        will_cancel=will_cancel,
        cancellation_probability=round(probability, 4),
        risk_level=risk_level,
        estimated_revenue_at_risk=estimated_revenue_at_risk,
        model_used=a.model_info.get("model_name", "unknown"),
    )
