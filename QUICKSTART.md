# mlship Quick Start Guide

Get your ML models serving in minutes.

## Installation

### Prerequisites

**Python Version Requirements:**
- ✅ **Python 3.11 or 3.12** (Recommended)
- ❌ **Python 3.13** does NOT work (PyTorch and TensorFlow don't support it yet)
- ❌ **Python 3.10 and below** not recommended (limited support)

### Step 1: Check Your Python Version

```bash
# Check your Python version
python3 --version
```

**If you have Python 3.13:**
You need to install Python 3.12 or 3.11.

**macOS:**
```bash
# Install Python 3.12 using Homebrew
brew install python@3.12

# Verify installation
python3.12 --version
```

**Linux (Ubuntu/Debian):**
```bash
# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.12
sudo apt install python3.12 python3.12-venv

# Verify installation
python3.12 --version
```

**Windows:**
Download Python 3.12 from [python.org/downloads](https://www.python.org/downloads/) and install it.

### Step 2: Create Virtual Environment

**Always use a virtual environment** to avoid conflicts with other Python projects:

```bash
# Create test env with Python 3.12
  mkdir ~/test-mlship && cd ~/test-mlship
  python3.12 -m venv test_env

# Activate it
  source test_env/bin/activate

  # Verify it's Python 3.12
python --version  # Should show Python 3.12.x



# Activate the virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

After activation, you should see `(venv)` in your terminal prompt.

### Step 3: Install mlship

```bash
# Basic installation (for sklearn models)
pip install mlship

# For HuggingFace models (includes PyTorch)
pip install 'mlship[huggingface]'

# For all frameworks
pip install 'mlship[all]'
```

**Notes:**
- On **zsh** (macOS default), you must quote the brackets: `'mlship[huggingface]'`
- **Automatic version check**: If you try to install with Python 3.13, pip will automatically reject the installation with a clear error message. This is expected - just use Python 3.11 or 3.12 as shown in Step 1.

### Step 4: Verify Installation

```bash
# Check mlship version
mlship --version

# Should output: mlship version 0.1.x

# Check help
mlship --help
```

**Test with a simple model:**

```bash
# Install scikit-learn
pip install scikit-learn

# Create a test model
python -c "
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, 'test_model.pkl')
print('✅ Test model created: test_model.pkl')
"

# Serve the model
mlship serve test_model.pkl
```

Visit http://localhost:8000/docs to see the interactive API documentation.

**If everything works, you're ready to go!** 🎉

---

## HuggingFace Hub Models

The fastest way to try mlship is with HuggingFace Hub models (no model files needed).

⚠️ **IMPORTANT**: HuggingFace models require PyTorch. If you only installed `pip install mlship`, you'll get an error. Install the huggingface extras:

```bash
pip install 'mlship[huggingface]'
```

This installs transformers + PyTorch which are required for HuggingFace models.

### Example 1: Sentiment Analysis

**Serve:**
```bash
mlship serve distilbert-base-uncased-finetuned-sst-2-english --source huggingface
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "This product is amazing!"}'
```

**Expected Response:**
```json
{
  "prediction": "POSITIVE",
  "probability": 0.9998,
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

**Try negative sentiment:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "This is terrible and disappointing"}'
```

**Expected Response:**
```json
{
  "prediction": "NEGATIVE",
  "probability": 0.9997,
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

---

### Example 2: Text Generation (GPT-2)

**Serve:**
```bash
mlship serve gpt2 --source huggingface
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "Once upon a time"}'
```

**Expected Response:**
```json
{
  "prediction": "Once upon a time, the world was a place of great beauty...",
  "model_name": "gpt2"
}
```

**Try a different prompt:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": "The future of AI is"}'
```

---

## Local Model Examples

### Example 3: Scikit-learn Model

**Create and train a model:**

```python
# train_sklearn_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

# Create sample data
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'sklearn_model.pkl')
print('✅ Model saved to sklearn_model.pkl')
```

```bash
python train_sklearn_model.py
```

**Serve:**
```bash
mlship serve sklearn_model.pkl
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.5, 2.3, -0.5, 1.2]}'
```

**Expected Response:**
```json
{
  "prediction": 0,
  "probability": 0.87,
  "model_name": "sklearn_model"
}
```

---

### Example 4: PyTorch Model

**Create and train a model:**

```python
# train_pytorch_model.py
import torch
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

# Create and save model
model = SimpleModel()
model.eval()

# Save FULL model (not just state_dict)
torch.save(model, 'pytorch_model.pt')
print('✅ Model saved to pytorch_model.pt')
```

```bash
python train_pytorch_model.py
```

**Serve:**
```bash
mlship serve pytorch_model.pt
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

**Expected Response:**
```json
{
  "prediction": [0.234, -0.156],
  "model_name": "pytorch_model"
}
```

---

### Example 5: TensorFlow/Keras Model

**Create and train a model:**

```python
# train_tensorflow_model.py
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Create sample data
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

# Define model
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=10, verbose=0)

# Save model
model.save('tensorflow_model.h5')
print('✅ Model saved to tensorflow_model.h5')
```

```bash
python train_tensorflow_model.py
```

**Serve:**
```bash
mlship serve tensorflow_model.h5
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, -0.3, 0.8]}'
```

**Expected Response:**
```json
{
  "prediction": 0.6234,
  "model_name": "tensorflow_model"
}
```

---

## Additional Features

### Interactive API Documentation

Open your browser to `http://localhost:8000/docs` for automatic Swagger UI with:
- Try out API calls directly
- See request/response schemas
- Test different inputs

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded"
}
```

### Model Info

```bash
curl http://localhost:8000/info
```

**Response:**
```json
{
  "name": "sklearn_model",
  "framework": "sklearn",
  "type": "RandomForestClassifier",
  "input_features": 4
}
```

### Custom Port

```bash
mlship serve model.pkl --port 5000
```

### Model Name

```bash
mlship serve model.pkl --name "fraud-detector"
```

---

## Framework Support

| Framework | Install | Serve Command |
|-----------|---------|---------------|
| **scikit-learn** | `pip install mlship scikit-learn` | `mlship serve model.pkl` |
| **PyTorch** | `pip install mlship torch` | `mlship serve model.pt` |
| **TensorFlow** | `pip install mlship tensorflow` | `mlship serve model.h5` |
| **HuggingFace Hub** | `pip install mlship transformers` | `mlship serve model-id --source huggingface` |

Or install everything:
```bash
pip install mlship[all]
```

---

## Tips

1. **Start with HuggingFace Hub models** - No files needed, easiest way to try mlship
2. **Use small models first** - `distilbert-base-uncased-finetuned-sst-2-english` is great for testing (268MB)
3. **Check the docs** - `http://localhost:8000/docs` shows all endpoints and schemas
4. **Test with curl first** - Verify API works before integrating with your app
5. **Use --reload in dev** - `mlship serve model.pkl --reload` auto-restarts on code changes

---

## What's Next?

- **Production deployment?** See [CONTRIBUTING.md](CONTRIBUTING.md) for best practices
- **Why mlship?** Read [WHY_MLSHIP.md](WHY_MLSHIP.md) to understand how mlship compares to other tools
- **Custom pipelines?** Check the full [README.md](README.md) for advanced features
- **Found a bug?** Report it at [GitHub Issues](https://github.com/sudhanvalabs/mlship/issues)

---

## Troubleshooting

**"Module not found" error?**
```bash
# Make sure framework is installed
pip install transformers  # for HuggingFace
pip install torch         # for PyTorch
pip install tensorflow    # for TensorFlow
```

**Port already in use?**
```bash
mlship serve model.pkl --port 5000  # Try different port
```

**Model not loading?**
- For PyTorch: Save full model with `torch.save(model, 'model.pt')`, not state_dict
- For HuggingFace: Use `--source huggingface` flag for Hub models
- Check file exists: `ls -lh model.pkl`

**Getting old version after update?**

First, check if your installed version matches the latest on PyPI:

```bash
# Check your installed version
pip show mlship | grep Version

# Compare with latest on PyPI
# Visit: https://pypi.org/project/mlship/
```

If the versions don't match, clear pip's cache:

```bash
# Clear pip cache
pip cache purge

# Install without cache
pip install --no-cache-dir mlship

# Or explicitly install latest version
pip install --upgrade --no-cache-dir mlship

# Verify version
pip show mlship
```

This happens when PyPI releases a new version but pip uses cached data. The `--no-cache-dir` flag forces pip to check PyPI directly.

---

**Happy serving!** 🚀
