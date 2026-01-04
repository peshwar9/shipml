# ShipML

**Turn any ML model into a REST API with one command.**

```bash
shipml serve model.pkl
```

Deploy your machine learning models locally in seconds—no Docker, no YAML, no configuration files.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Features

- ✅ **One-command deployment** - No configuration needed
- ✅ **Works offline** - Zero internet dependency after installation
- ✅ **Auto-generated API** - REST API with interactive docs
- ✅ **Multi-framework** - Supports scikit-learn, PyTorch, TensorFlow, Hugging Face
- ✅ **Platform agnostic** - Works on macOS, Windows, and Linux
- ✅ **Fast** - Deploy in seconds, predictions in milliseconds
- ✅ **Production-ready** - Input validation, error handling, health checks

---

## Quick Start

### Installation

**macOS / Linux:**
```bash
# Using pip
pip install shipml

# Using uv (recommended - faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install shipml
```

**Windows:**
```powershell
# Using pip
pip install shipml

# Using uv (recommended)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv pip install shipml
```

### Your First API

```bash
# 1. Train a model (or use an existing one)
python
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.datasets import make_classification
>>> import joblib
>>> X, y = make_classification(n_samples=100, n_features=10, random_state=42)
>>> model = RandomForestClassifier()
>>> model.fit(X, y)
>>> joblib.dump(model, 'model.pkl')
>>> exit()

# 2. Serve it
shipml serve model.pkl

# 3. Test it (in another terminal)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}'

# Response:
# {"prediction": 0, "probability": 0.87, "model_name": "model"}
```

### View Interactive Docs

Open http://localhost:8000/docs in your browser for Swagger UI with interactive API testing.

---

## Supported Frameworks

| Framework | File Extensions | Example |
|-----------|----------------|---------|
| **Scikit-learn** | `.pkl`, `.joblib` | `shipml serve model.pkl` |
| **PyTorch** | `.pt`, `.pth` | `shipml serve model.pt` |
| **TensorFlow/Keras** | `.h5`, `.keras`, `SavedModel/` | `shipml serve model.h5` |
| **Hugging Face** | Model directory | `shipml serve sentiment-model/` |
| **XGBoost** | `.json`, `.pkl` | Coming soon |
| **LightGBM** | `.txt`, `.pkl` | Coming soon |

---

## Platform-Specific Instructions

### macOS

**Installation:**
```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ShipML
uv pip install shipml

# For specific frameworks
uv pip install shipml[sklearn]      # Scikit-learn
uv pip install shipml[pytorch]      # PyTorch
uv pip install shipml[tensorflow]   # TensorFlow
uv pip install shipml[huggingface]  # Hugging Face
uv pip install shipml[all]          # All frameworks
```

**Usage:**
```bash
shipml serve model.pkl
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

# Install ShipML
uv pip install shipml[all]
```

**RHEL/CentOS/Fedora:**
```bash
# Install Python
sudo dnf install python3 python3-pip

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ShipML
uv pip install shipml[all]
```

### Windows

**PowerShell (Administrator):**
```powershell
# Install Python 3.8+ from https://www.python.org/downloads/

# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install ShipML
uv pip install shipml[all]
```

**Usage:**
```powershell
shipml serve model.pkl
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
shipml serve model.pkl --port 5000
```

### Custom Host

```bash
shipml serve model.pkl --host 0.0.0.0  # Allow external connections
```

### Custom Model Name

```bash
shipml serve model.pkl --name "fraud-detector"
```

### Development Mode (Auto-reload)

```bash
shipml serve model.pkl --reload
```

---

## Hugging Face Models

ShipML supports Hugging Face Transformers for NLP tasks.

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
shipml serve sentiment-model/
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

Benchmark results for different model types:

| Model Type | Load Time | Prediction (avg) | Notes |
|------------|-----------|------------------|-------|
| **Scikit-learn** (10 features) | ~50ms | 2-5ms | Fastest |
| **PyTorch** (small CNN) | ~1s | 10-20ms | CPU inference |
| **TensorFlow** (Keras model) | ~2s | 15-30ms | CPU inference |
| **Hugging Face** (DistilBERT) | ~5-10s | 10-15ms | First prediction slower |

**Hardware:** MacBook Pro M1, 16GB RAM, CPU only

See [PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks.

---

## Examples

Check the `examples/` directory:

- `examples/sklearn_example.py` - Scikit-learn classification
- `examples/pytorch_example.py` - PyTorch neural network
- `examples/huggingface_example.py` - Sentiment analysis with BERT

```bash
# Run an example
python examples/sklearn_example.py
shipml serve fraud_detector.pkl
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

### "Command not found: shipml"

Make sure ShipML is installed and in your PATH:

```bash
pip install shipml
# or
uv pip install shipml

# Verify installation
shipml --version
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
shipml serve model.pkl --port 8001
```

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

## Support

- 📖 [Documentation](https://github.com/prabhueshwarla/shipml#readme)
- 🐛 [Report Issues](https://github.com/prabhueshwarla/shipml/issues)
- 💬 [Discussions](https://github.com/prabhueshwarla/shipml/discussions)

---

## Roadmap

- [x] Scikit-learn support
- [x] PyTorch support
- [x] TensorFlow support
- [x] Hugging Face support
- [ ] XGBoost support
- [ ] LightGBM support
- [ ] GPU inference
- [ ] Batch prediction optimization
- [ ] Model preprocessing pipelines
- [ ] Docker deployment option

---

**Made with ❤️ for the ML community**
