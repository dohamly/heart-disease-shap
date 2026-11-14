"""
schemas.py
----------
Pydantic models validating the requests and responses of the API.
"""

from pydantic import BaseModel, ConfigDict, Field


class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0 = female, 1 = male")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=60, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0/1)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (0/1)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels coloured by fluoroscopy (0-3)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 58, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250,
                "fbs": 0, "restecg": 0, "thalach": 148, "exang": 0,
                "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 2,
            }
        }
    )


class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    probability_disease: float
    probability_no_disease: float
    model_name: str


class FeatureContribution(BaseModel):
    feature: str
    value: float


class ExplanationOutput(BaseModel):
    prediction: PredictionOutput
    contributions: list[FeatureContribution]          # one per encoded column (one-hot)
    variable_contributions: list[FeatureContribution]  # one per clinical variable (sum of its encoded columns)
    base_value: float
    base_probability: float                            # predicted probability of the average patient
    output_space: str                                  # "log-odds" or "probability": unit of base_value and of the contributions
