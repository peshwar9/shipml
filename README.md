# mlship

**Turn any ML model into a REST API with one command.**

```bash
mlship serve model.pkl
```

Deploy your machine learning models locally in seconds—no Docker, no YAML, no configuration files.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Why mlship?** See our [positioning document](WHY_MLSHIP.md) for how mlship compares to transformers-serve, vLLM, Ollama, and BentoML.

---

## Features

- ✅ **One-command deployment** - No configuration needed
- ✅ **Works offline** - Zero internet dependency after installation
- ✅ **Auto-generated API** - REST API with interactive docs
- ✅ **Multi-framework** - Supports scikit-learn, PyTorch, TensorFlow, **and HuggingFace** (local + Hub)
- ✅ **HuggingFace Hub integration** - Serve models directly from Hub without downloading
- ✅ **Platform agnostic** - Works on macOS, Windows, and Linux
- ✅ **Fast** - Deploy in seconds, predictions in milliseconds
- ✅ **Demo-ready** - Input validation, error handling, health checks for local testing

---

## Quick Start

### Installation

**macOS / Linux:**
```bash
# Using pip (recommended)
pip install mlship

# Using uv (faster alternative)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install mlship
```

**Windows:**
```powershell
# Using pip (recommended)
pip install mlship

# Using uv (faster alternative)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv pip install mlship
```

### Your First API

**Step 1: Install dependencies**

```bash
# Install mlship and scikit-learn
pip install mlship scikit-learn joblib
```

**Step 2: Create and train a model**

Create a file `train_model.py`:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

# Train a simple model
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save it
joblib.dump(model, 'model.pkl')
print('✅ Model saved to model.pkl')
```

Run it:
```bash
python train_model.py
```

**Step 3: Serve it**

```bash
mlship serve model.pkl
```

You'll see:
```
🚀 mlship
   Loading model: model.pkl

🔍 Detecting framework... ✓ sklearn
📦 Loading model... ✓ Success

📊 Model Information:
   Name:       model
   Framework:  sklearn
   Features:   4

🎉 Server starting!
   URL:      http://127.0.0.1:8000
   API Docs: http://127.0.0.1:8000/docs
```

**Step 4: Test it** (in another terminal)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

Response:
```json
{"prediction": 0, "probability": 0.87, "model_name": "model"}
```

### View Interactive Docs

Open http://localhost:8000/docs in your browser for Swagger UI with interactive API testing.

---

## Supported Frameworks

| Framework | File Extensions / Source | Example |
|-----------|-------------------------|---------|
| **Scikit-learn** | `.pkl`, `.joblib` | `mlship serve model.pkl` |
| **PyTorch** | `.pt`, `.pth` | `mlship serve model.pt` |
| **TensorFlow/Keras** | `.h5`, `.keras`, `SavedModel/` | `mlship serve model.h5` |
| **Hugging Face (Local)** | Model directory | `mlship serve sentiment-model/` |
| **Hugging Face (Hub)** | Model ID | `mlship serve bert-base-uncased --source huggingface` |
| **XGBoost** | `.json`, `.pkl` | Coming soon |
| **LightGBM** | `.txt`, `.pkl` | Coming soon |

---

## HuggingFace Hub Support

Serve models directly from HuggingFace Hub without downloading them first:

```bash
# Serve a text classification model
mlship serve distilbert-base-uncased-finetuned-sst-2-english --source huggingface

# Serve GPT-2 for text generation
mlship serve gpt2 --source huggingface --port 5000

# Serve any HuggingFace model
mlship serve <model-id> --source huggingface
```

**What happens:**
- mlship downloads the model on first use (with progress bars)
- Model is cached locally in `~/.cache/huggingface/` for faster subsequent loads
- No manual model download or setup required

**Test it:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "This product is amazing!"}'
```

Response:
```json
{
  "prediction": "POSITIVE",
  "probability": 0.9998,
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

**Note:** You need transformers installed:
```bash
pip install mlship[huggingface]
# or
pip install mlship transformers
```

---

## Platform-Specific Instructions

### macOS

**Installation:**
```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install mlship
uv pip install mlship

# For specific frameworks
uv pip install mlship[sklearn]      # Scikit-learn
uv pip install mlship[pytorch]      # PyTorch
uv pip install mlship[tensorflow]   # TensorFlow
uv pip install mlship[huggingface]  # Hugging Face
uv pip install mlship[all]          # All frameworks
```

**Usage:**
```bash
mlship serve model.pkl
# Server runs at http://127.0.0.1:8000
```

### Linux

**Ubuntu/Debian:**
```bash
# Install Python 3.8+ (if not already installed)
sudo apt update
sudo apt install python3 python3-pip

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install mlship
uv pip install mlship[all]
```

**RHEL/CentOS/Fedora:**
```bash
# Install Python
sudo dnf install python3 python3-pip

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install mlship
uv pip install mlship[all]
```

### Windows

**PowerShell (Administrator):**
```powershell
# Install Python 3.8+ from https://www.python.org/downloads/

# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install mlship
uv pip install mlship[all]
```

**Usage:**
```powershell
mlship serve model.pkl
# Server runs at http://127.0.0.1:8000
```

---

## API Reference

Every deployed model automatically gets:

### Endpoints

#### `POST /predict`
Make predictions with your model.

**Request:**
```json
{
  "features": [1.0, 2.0, 3.0, 4.0]
}
```

**Response (Classification):**
```json
{
  "prediction": 0,
  "probability": 0.87,
  "model_name": "fraud_detector"
}
```

**Response (Regression):**
```json
{
  "prediction": 42.5,
  "model_name": "price_predictor"
}
```

#### `GET /health`
Check server health.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "fraud_detector",
  "uptime_seconds": 3600
}
```

#### `GET /info`
Get model metadata.

**Response:**
```json
{
  "model_name": "fraud_detector",
  "model_type": "sklearn.ensemble.RandomForestClassifier",
  "framework": "scikit-learn",
  "input_features": 10,
  "output_type": "classification"
}
```

#### `GET /docs`
Interactive API documentation (Swagger UI).

---

## Advanced Usage

### Custom Port

```bash
mlship serve model.pkl --port 5000
```

### Custom Host

```bash
mlship serve model.pkl --host 0.0.0.0  # Allow external connections
```

### Custom Model Name

```bash
mlship serve model.pkl --name "fraud-detector"
```

### Development Mode (Auto-reload)

```bash
mlship serve model.pkl --reload
```

### Custom Preprocessing/Postprocessing Pipelines

**What are pipelines?**

By default, mlship expects models to receive data in a standard format and returns predictions as-is. But real-world ML deployments often need:
- **Preprocessing**: Normalization, feature engineering, text extraction, scaling
- **Postprocessing**: Custom formatting, thresholds, business logic, metadata

Pipelines let you inject custom logic before and after model prediction without modifying mlship's code.

**How it works:**

```
Without Pipeline:
User Request → Validate → Model.predict() → Response

With Pipeline:
User Request → Pipeline.preprocess() → Model.predict() → Pipeline.postprocess() → Response
```

**Why it differs by framework:**

| Framework | Input Type | Preprocessing Example | Notes |
|-----------|-----------|----------------------|-------|
| **Sklearn** | Numpy arrays | Normalization, scaling | Model is simple `.predict()` method |
| **PyTorch** | Tensors | Custom transforms, device placement | Model is callable |
| **HuggingFace** | Raw text | Extract text from request | Pipeline handles tokenization internally! |
| **TensorFlow** | Tensors | Reshaping, normalization | Model is callable |

**Example 1: Sklearn with normalization**

```python
# my_pipeline.py
from mlship.pipeline import Pipeline
import numpy as np

class NormalizationPipeline(Pipeline):
    def __init__(self, model_path):
        super().__init__(model_path)
        # Load normalization parameters
        self.mean = np.array([50.0, 100.0])
        self.std = np.array([10.0, 20.0])

    def preprocess(self, request_data):
        """Normalize features before prediction"""
        features = np.array(request_data["features"])
        return (features - self.mean) / self.std

    def postprocess(self, model_output):
        """Add metadata to response"""
        model_output["preprocessing"] = "normalized"
        return model_output
```

**Example 2: HuggingFace text processing**

```python
# sentiment_pipeline.py
from mlship.pipeline import Pipeline

class SentimentPipeline(Pipeline):
    def preprocess(self, request_data):
        """Extract text - HuggingFace pipeline handles tokenization!"""
        return request_data.get("text", "")

    def postprocess(self, model_output):
        """Format output nicely"""
        result = model_output[0] if isinstance(model_output, list) else model_output
        return {
            "sentiment": result["label"],
            "confidence": round(result["score"], 4)
        }
```

**Usage:**

```bash
# Serve with custom pipeline
mlship serve model.pkl --pipeline my_pipeline.NormalizationPipeline

# Test with custom input format
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [100, 200]}'
```

**Real-world use cases:**
- 🔤 Text extraction and custom formatting for NLP models
- 📊 Feature normalization matching training-time preprocessing
- 🎯 Business logic (thresholds, alerts, A/B testing)
- 🔄 Multi-model ensembles with voting logic
- 📝 Logging, monitoring, and audit trails

**See working examples:**
- `examples/pipelines/sentiment_pipeline.py` - HuggingFace text processing
- `examples/pipelines/sklearn_normalization.py` - Feature normalization
- `examples/pipelines/simple_test.py` - Minimal template

---

## Hugging Face Models

mlship supports Hugging Face Transformers for NLP tasks.

### Download and Serve

```python
# download_model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./sentiment-model")
tokenizer.save_pretrained("./sentiment-model")
```

```bash
# Serve it
mlship serve sentiment-model/
```

### Test with Text

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "This product is amazing!"}'

# Response:
# {"prediction": "POSITIVE", "probability": 0.9999, "model_name": "sentiment-model"}
```

**Supported Tasks:**
- Text classification (sentiment, topic, etc.)
- Named Entity Recognition (NER)
- Question Answering
- Text generation
- Summarization

---

## Performance

### Benchmarks

| Model Type | Load Time | Prediction (avg) | Notes |
|------------|-----------|------------------|-------|
| **Scikit-learn** (10 features) | ~50ms | 2-5ms | Fastest |
| **PyTorch** (small CNN) | ~1s | 10-20ms | CPU inference |
| **TensorFlow** (Keras model) | ~2s | 15-30ms | CPU inference |
| **Hugging Face** (DistilBERT) | ~5-10s | 10-15ms | First prediction slower |

**Hardware:** MacBook Pro M1, 16GB RAM, CPU only

### Built-in Optimizations

mlship includes sensible defaults that work out-of-the-box:
- ✅ **Batch processing** - All loaders support batch inputs efficiently
- ✅ **Model eval mode** - PyTorch models loaded with `model.eval()`
- ✅ **CPU-optimized** - HuggingFace uses CPU-optimized pipelines
- ✅ **Framework defaults** - Leverages sklearn, PyTorch, TensorFlow optimizations

**Note:** mlship is "demo-ready," not production-ready. We intentionally avoid advanced optimizations (quantization, GPU acceleration, advanced batching) to maintain simplicity. For production deployments, consider tools like TorchServe, vLLM, or Triton.

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and when to graduate to production tools.

---

## Examples

Check the `examples/` directory:

- `examples/sklearn_example.py` - Scikit-learn classification
- `examples/pytorch_example.py` - PyTorch neural network
- `examples/huggingface_example.py` - Sentiment analysis with BERT

```bash
# Run an example
python examples/sklearn_example.py
mlship serve fraud_detector.pkl
```

---

## Use Cases

### For Students
- ✅ Demo ML projects in presentations
- ✅ Test models interactively
- ✅ Share models with classmates
- ✅ No cloud bills or sleeping containers

### For Data Scientists
- ✅ Quick model validation
- ✅ Prototype APIs before production
- ✅ Share models with stakeholders
- ✅ Test different model versions

### For Educators
- ✅ Teach API concepts without DevOps complexity
- ✅ Students focus on ML, not deployment
- ✅ Works offline in classrooms
- ✅ No infrastructure setup needed

---

## Troubleshooting

### "Command not found: mlship"

Make sure mlship is installed and in your PATH:

```bash
pip install mlship
# or
uv pip install mlship

# Verify installation
mlship --version
```

### "Framework not detected"

Ensure your model file has the correct extension:

```bash
# Scikit-learn
joblib.dump(model, 'model.pkl')  # or .joblib

# PyTorch
torch.save(model, 'model.pt')    # or .pth

# TensorFlow
model.save('model.h5')           # or .keras
```

### "Invalid input shape"

Check your model's expected input features:

```bash
curl http://localhost:8000/info
# Look at "input_features" in the response
```

### Port already in use

Use a different port:

```bash
mlship serve model.pkl --port 8001
```

---

## Development

### Running Tests

```bash
# Run all tests
.venv/bin/python -m pytest

# Run pipeline tests only
.venv/bin/python -m pytest tests/test_pipelines.py -v

# Run specific framework tests
.venv/bin/python -m pytest tests/test_pipelines.py::TestSklearnPipeline -v
.venv/bin/python -m pytest tests/test_pipelines.py::TestPyTorchPipeline -v
.venv/bin/python -m pytest tests/test_pipelines.py::TestHuggingFacePipeline -v

# Run single test
.venv/bin/python -m pytest tests/test_pipelines.py::TestSklearnPipeline::test_sklearn_pipeline_preprocess -v
```

**Useful options:**
- `-v` - Verbose output
- `-s` - Show print statements
- `-x` - Stop on first failure
- `--no-cov` - Skip coverage report (faster)
- `--lf` - Run only failed tests from last run

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Running tests
- Code style guidelines
- Submitting pull requests

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Documentation

mlship includes focused documentation for different audiences:

| Document | Audience | What's Inside |
|----------|----------|---------------|
| **[README.md](README.md)** | End users | Installation, quick start, usage examples, API reference |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Contributors | Development setup, testing, code style, PR process |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Developers | Design philosophy, technical decisions, when to use mlship vs production tools |
| **[PUBLISHING.md](PUBLISHING.md)** | Maintainers | How to publish releases to PyPI (internal use) |

**New to mlship?** Start with the Quick Start section above.
**Want to contribute?** Read [CONTRIBUTING.md](CONTRIBUTING.md).
**Curious about design?** Check [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Support

- 📖 [Documentation](https://github.com/prabhueshwarla/mlship#readme)
- 🐛 [Report Issues](https://github.com/prabhueshwarla/mlship/issues)
- 💬 [Discussions](https://github.com/prabhueshwarla/mlship/discussions)

---

## Roadmap

- [x] Scikit-learn support
- [x] PyTorch support
- [x] TensorFlow support
- [x] Hugging Face support
- [x] Cross-platform CI/CD testing
- [x] Custom preprocessing/postprocessing pipelines
- [ ] XGBoost support
- [ ] LightGBM support
- [ ] GPU inference
- [ ] Batch prediction optimization
- [ ] Docker deployment option

---

**Made with ❤️ for the ML community**
