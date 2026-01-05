"""Hugging Face Transformers model loader.

IMPLEMENTATION STATUS: Skeleton only - not fully implemented yet.
Add this to Phase 1 if Hugging Face support is critical for launch.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

from shipml.errors import ModelLoadError, ValidationError
from shipml.loaders.base import ModelLoader


class HuggingFaceLoader(ModelLoader):
    """
    Loader for Hugging Face Transformers models.

    Supports:
    - Text classification (BERT, RoBERTa, DistilBERT)
    - Text generation (GPT-2, GPT-Neo)
    - Named Entity Recognition
    - Question Answering
    - Summarization
    - Translation

    Model format: Directory with config.json, pytorch_model.bin, tokenizer files
    """

    def load(self, model_path: Path) -> Any:
        """
        Load Hugging Face model from directory.

        Args:
            model_path: Path to model directory (must contain config.json)

        Returns:
            Tuple of (pipeline, task_type)

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        try:
            from transformers import pipeline, AutoConfig

            # Check if directory has required files
            if not model_path.is_dir():
                raise ModelLoadError(
                    "Hugging Face models must be directories.\n\n"
                    "Download a model first:\n"
                    "  from transformers import AutoModel, AutoTokenizer\n"
                    "  model = AutoModel.from_pretrained('bert-base-uncased')\n"
                    "  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\n"
                    "  model.save_pretrained('my-model/')\n"
                    "  tokenizer.save_pretrained('my-model/')\n\n"
                    "Then serve:\n"
                    "  shipml serve my-model/"
                )

            config_file = model_path / "config.json"
            if not config_file.exists():
                raise ModelLoadError(
                    f"Missing config.json in {model_path}\n\n"
                    "This doesn't look like a Hugging Face model directory."
                )

            # Detect task from config or directory name
            config = AutoConfig.from_pretrained(str(model_path))
            task_type = self._detect_task(config, model_path)

            # Load pipeline
            pipe = pipeline(task_type, model=str(model_path), device=-1)  # type: ignore[call-overload]

            return {"pipeline": pipe, "task": task_type}

        except ImportError:
            raise ModelLoadError(
                "Hugging Face Transformers is not installed.\n\n"
                "Install with:\n"
                "  uv pip install transformers\n"
                "Or:\n"
                "  uv pip install shipml[huggingface]"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load Hugging Face model: {e}")

    def predict(self, model: Any, features: Any) -> Dict[str, Any]:
        """
        Run prediction with Hugging Face pipeline.

        Args:
            model: Loaded model dict (pipeline + task)
            features: Input text (string or list of strings)

        Returns:
            Prediction results
        """
        pipeline_obj = model["pipeline"]
        task = model["task"]

        # Convert features to appropriate format
        if isinstance(features, list) and len(features) > 0:
            if isinstance(features[0], str):
                # List of strings
                inputs = features
            else:
                # Numeric features - not supported for transformers
                raise ValidationError(
                    "Hugging Face models expect text input, not numeric features.\n\n"
                    "Example for text classification:\n"
                    '{"features": "This is a great product!"}\n\n'
                    "Example for batch:\n"
                    '{"features": ["Text 1", "Text 2"]}'
                )
        elif isinstance(features, str):
            # Single string
            inputs = features  # type: ignore[assignment]
        else:
            raise ValidationError(
                "Invalid input format for Hugging Face model.\n\n"
                "Expected: string or list of strings\n"
                f"Received: {type(features)}"
            )

        # Run pipeline
        results = pipeline_obj(inputs)

        # Format output based on task
        return self._format_output(results, task)

    def get_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract model metadata."""
        pipeline_obj = model["pipeline"]
        task = model["task"]

        metadata = {
            "model_type": pipeline_obj.model.__class__.__name__,
            "framework": "huggingface-transformers",
            "task": task,
            "output_type": self._get_output_type(task),
        }

        # Note: Don't include model_name here - it's added by the server

        return metadata

    def validate_input(self, model: Any, features: Union[str, List[str]]) -> None:
        """Validate input format."""
        if not isinstance(features, (str, list)):
            raise ValidationError(
                "Hugging Face models require text input (string or list of strings).\n\n"
                f"Expected: str or List[str]\n"
                f"Received: {type(features)}"
            )

        if isinstance(features, list):
            if not all(isinstance(item, str) for item in features):
                raise ValidationError(
                    "All items in features list must be strings.\n\n"
                    'Example: {"features": ["Text 1", "Text 2"]}'
                )

    # Helper methods

    def _detect_task(self, config: Any, model_path: Path) -> str:
        """Detect task type from config or directory name."""
        # Try to infer from model architecture
        arch = config.architectures[0] if config.architectures else ""

        if "ForSequenceClassification" in arch:
            return "text-classification"
        elif "ForTokenClassification" in arch:
            return "token-classification"
        elif "ForQuestionAnswering" in arch:
            return "question-answering"
        elif "ForCausalLM" in arch or "GPT" in arch:
            return "text-generation"
        elif "ForSeq2SeqLM" in arch:
            # Could be summarization or translation
            return "summarization"
        else:
            # Default to text-classification
            return "text-classification"

    def _format_output(self, results: Any, task: str) -> Dict[str, Any]:
        """Format pipeline output to standard format."""
        if task == "text-classification":
            if isinstance(results, list):
                # Batch results
                result = results[0]  # Take first for now
            else:
                result = results

            return {
                "prediction": result["label"],
                "probability": result["score"],
            }

        elif task == "text-generation":
            if isinstance(results, list):
                result = results[0]
            else:
                result = results

            return {
                "prediction": result["generated_text"],
            }

        elif task == "token-classification":
            # NER results
            return {
                "prediction": results,
            }

        elif task == "question-answering":
            return {
                "prediction": results["answer"],
                "probability": results["score"],
            }

        else:
            # Generic format
            return {"prediction": results}

    def _get_output_type(self, task: str) -> str:
        """Get output type from task."""
        classification_tasks = ["text-classification", "question-answering"]
        if task in classification_tasks:
            return "classification"
        else:
            return "generation"
