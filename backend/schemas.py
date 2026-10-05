"""
Pydantic schemas for the BookPulse hotel booking cancellation API.

The request schema mirrors the RAW columns of the Hotel Booking Demand
dataset (minus `company`/`agent`, which were dropped entirely in
Stage 4, and minus the leakage columns `reservation_status` /
`reservation_status_date`, which never existed as model features).

Literal fields encode the known category sets from the public
dataset, so FastAPI/Swagger renders them as dropdowns and rejects an
unrecognised category with a clear 422 error -- satisfying the
"accept and validate user inputs" / "handle invalid input
appropriately" requirements for Stage 9.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


Hotel = Literal["Resort Hotel", "City Hotel"]
Meal = Literal["BB", "FB", "HB", "SC", "Undefined"]
MarketSegment = Literal[
    "Direct", "Corporate", "Online TA", "Offline TA/TO",
    "Complementary", "Groups", "Undefined", "Aviation",
]
DistributionChannel = Literal["Direct", "Corporate", "TA/TO", "Undefined", "GDS"]
RoomType = Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
DepositType = Literal["No Deposit", "Refundable", "Non Refund"]
CustomerType = Literal["Transient", "Contract", "Transient-Party", "Group"]
ArrivalMonth = Literal[
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class BookingRequest(BaseModel):
    hotel: Hotel = Field(..., description="Which property the booking is for")

    lead_time: int = Field(..., ge=0, le=800, description="Days between booking date and arrival date")
    arrival_date_year: int = Field(..., ge=2015, le=2100)
    arrival_date_month: ArrivalMonth
    arrival_date_week_number: int = Field(..., ge=1, le=53)
    arrival_date_day_of_month: int = Field(..., ge=1, le=31)

    stays_in_weekend_nights: int = Field(..., ge=0, le=30)
    stays_in_week_nights: int = Field(..., ge=0, le=60)

    adults: int = Field(..., ge=0, le=10)
    children: int = Field(0, ge=0, le=10)
    babies: int = Field(0, ge=0, le=10)

    meal: Meal
    country: str = Field(..., min_length=2, max_length=3, description="ISO country code, e.g. 'PRT', 'GBR'")
    market_segment: MarketSegment
    distribution_channel: DistributionChannel

    is_repeated_guest: Literal[0, 1] = 0
    previous_cancellations: int = Field(0, ge=0, le=100)
    previous_bookings_not_canceled: int = Field(0, ge=0, le=100)

    reserved_room_type: RoomType
    assigned_room_type: RoomType
    booking_changes: int = Field(0, ge=0, le=30)
    deposit_type: DepositType
    customer_type: CustomerType

    adr: float = Field(..., ge=0, le=10000, description="Average Daily Rate, in the hotel's currency")
    required_car_parking_spaces: int = Field(0, ge=0, le=10)
    total_of_special_requests: int = Field(0, ge=0, le=10)

    @field_validator("country")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def check_booking_is_physically_valid(self) -> "BookingRequest":
        """
        Mirrors the data-quality rules established during Stage 3/4:
        a booking with no guests at all, or no nights at all, is not a
        valid reservation and is rejected here rather than silently
        passed on to the model.
        """
        total_guests = self.adults + self.children + self.babies
        if total_guests == 0:
            raise ValueError("A booking must have at least one guest (adults + children + babies > 0).")

        total_nights = self.stays_in_weekend_nights + self.stays_in_week_nights
        if total_nights == 0:
            raise ValueError("A booking must include at least one night (weekend or weekday).")

        return self

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "total_of_special_requests": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: Literal["Cancelled", "Not Cancelled"]
    will_cancel: bool
    cancellation_probability: float = Field(..., ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]
    estimated_revenue_at_risk: float = Field(
        ..., description="cancellation_probability * adr * total nights -- expected revenue exposure, not a guaranteed loss"
    )
    model_used: str


class ModelInfoResponse(BaseModel):
    model_name: str
    test_roc_auc: float
    test_f1: float
    test_recall: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
