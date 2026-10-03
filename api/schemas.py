"""
schemas.py
----------
Modèles Pydantic pour valider les requêtes/réponses de l'API.
"""

from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age en années")
    sex: int = Field(..., ge=0, le=1, description="0 = Femme, 1 = Homme")
    cp: int = Field(..., ge=0, le=3, description="Type de douleur thoracique (0-3)")
    trestbps: float = Field(..., ge=60, le=250, description="Tension artérielle au repos")
    chol: float = Field(..., ge=80, le=700, description="Cholestérol sérique (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Glycémie à jeun > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Résultats ECG au repos")
    thalach: float = Field(..., ge=50, le=250, description="Fréquence cardiaque maximale")
    exang: int = Field(..., ge=0, le=1, description="Angine induite par l'effort")
    oldpeak: float = Field(..., ge=0, le=10, description="Dépression du segment ST")
    slope: int = Field(..., ge=0, le=2, description="Pente du segment ST")
    ca: int = Field(..., ge=0, le=3, description="Nombre de vaisseaux majeurs colorés")
    thal: int = Field(..., ge=0, le=3, description="Thalassémie")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 58, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250,
                "fbs": 0, "restecg": 0, "thalach": 148, "exang": 0,
                "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 2,
            }
        }


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
    contributions: list[FeatureContribution]
    base_value: float
