"""Pydantic models for request/response validation."""

from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request model for predictions (numeric models: sklearn, pytorch, tensorflow)."""

    features: Union[List[float], List[List[float]], Dict[str, Any]] = Field(
        ...,
        description="Numeric input features for prediction",
        examples=[[1.0, 2.0, 3.0, 4.0]],
    )


class PredictResponse(BaseModel):
    """Response model for predictions."""

    prediction: Union[int, float, str, List[float]]
    probability: Union[float, None] = None
    probabilities: Union[Dict[str, float], None] = None
    model_name: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    model_name: str
    uptime_seconds: int


class InfoResponse(BaseModel):
    """Response model for model info."""

    model_name: str
    model_type: str
    framework: str
    input_features: Union[int, None] = None
    output_type: Union[str, None] = None
    classes: Union[List[Any], None] = None


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str
    message: str
    details: Union[Dict[str, Any], None] = None
