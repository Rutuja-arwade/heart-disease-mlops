"""Request/response contracts for the prediction API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

EXAMPLE_PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 4,
    "trestbps": 150,
    "chol": 286,
    "fbs": 0,
    "restecg": 2,
    "thalach": 108,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 3,
    "thal": 7,
}


class PatientFeatures(BaseModel):
    """The 13 clinical attributes of the UCI Cleveland dataset."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_PATIENT})

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=70, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=650, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LVH")
    thalach: float = Field(..., ge=60, le=230, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=7, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: Optional[Literal[0, 1, 2, 3]] = Field(None, description="Major vessels coloured by fluoroscopy (optional)")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed defect, 7 reversible (optional)")

    @field_validator("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", mode="before")
    @classmethod
    def integral_codes(cls, value):
        """Accept 1.0 / "1" style codes (common when rows come from CSV) as integers."""
        if isinstance(value, str) and value.strip().lstrip("-").replace(".", "", 1).isdigit():
            value = float(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not predicted")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float = Field(..., description="Probability of heart disease (class 1)")
    risk_level: Literal["low", "moderate", "high"]
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_version: str
    count: int
    predictions: list[dict]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float
