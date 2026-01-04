"""FastAPI server generator for ML models."""

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from shipml.loaders.base import ModelLoader
from shipml.models import PredictRequest, HealthResponse, InfoResponse
from shipml.errors import ValidationError


def create_app(model: Any, loader: ModelLoader, model_name: str) -> FastAPI:
    """
    Dynamically generate FastAPI app for the model.

    Args:
        model: Loaded model object
        loader: Model loader instance
        model_name: Display name for the model

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title=f"{model_name} API",
        description=f"Auto-generated API for {model_name}. Built with ShipML.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Track startup time
    startup_time = time.time()

    # Get model metadata once at startup
    metadata = loader.get_metadata(model)

    # Create framework-specific request model
    framework = metadata.get("framework", "")
    if framework == "huggingface-transformers":
        # Text-based input for HuggingFace models
        from pydantic import BaseModel, Field
        from typing import Union, List

        class PredictRequest(BaseModel):
            features: Union[str, List[str]] = Field(
                ...,
                description="Text input for prediction",
                examples=["This product is amazing!"]
            )
    else:
        # Numeric input for sklearn/pytorch/tensorflow
        from shipml.models import PredictRequest

    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint - redirect to docs."""
        return {
            "message": f"Welcome to {model_name} API",
            "docs": "/docs",
            "health": "/health",
            "info": "/info",
        }

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """
        Check if the server is healthy and the model is loaded.

        Returns health status and uptime information.
        """
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=model_name,
            uptime_seconds=int(time.time() - startup_time),
        )

    @app.get("/info", response_model=InfoResponse, tags=["System"])
    async def model_info():
        """
        Get model metadata and configuration.

        Returns information about the model type, framework, and expected inputs.
        """
        return InfoResponse(model_name=model_name, **metadata)

    @app.post("/predict", tags=["Predictions"])
    async def predict(request: PredictRequest):
        """
        Make predictions using the model.

        Send input features as a JSON array and receive predictions.

        **Example request:**
        ```json
        {
          "features": [1.0, 2.0, 3.0, 4.0]
        }
        ```

        **Example response (classification):**
        ```json
        {
          "prediction": 0,
          "probability": 0.87,
          "model_name": "fraud_detector"
        }
        ```

        **Example response (regression):**
        ```json
        {
          "prediction": 42.5,
          "model_name": "price_predictor"
        }
        ```
        """
        try:
            # Validate input
            loader.validate_input(model, request.features)

            # Run prediction
            result = loader.predict(model, request.features)

            # Add model name to response
            result["model_name"] = model_name

            return result

        except ValidationError as e:
            # Return 400 for validation errors
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Validation Error",
                    "message": str(e),
                },
            )
        except Exception as e:
            # Return 500 for unexpected errors
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Prediction Failed",
                    "message": f"An error occurred during prediction: {str(e)}",
                },
            )

    # Add custom exception handler for better error messages
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail},
        )

    return app
