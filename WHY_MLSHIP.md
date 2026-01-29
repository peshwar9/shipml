# Why mlship?

## The Universal ML Model Server

mlship is the **only tool that serves sklearn, PyTorch, TensorFlow, and HuggingFace models with zero code**. One command, any framework, instant API.

## The Problem

As an ML engineer or data scientist, you've probably experienced this:

- Train a sklearn model → Learn how to serve it with Flask
- Move to PyTorch → Learn TorchServe or build custom FastAPI
- Experiment with transformers → Learn transformers-serve or Gradio
- Try TensorFlow → Learn TF Serving

**Each framework requires learning a different serving tool.** You spend more time on deployment than on modeling.

## The mlship Solution

**One tool. All frameworks. Zero code.**

```bash
# sklearn model
mlship serve fraud_model.pkl

# PyTorch model
mlship serve image_classifier.pt

# TensorFlow model
mlship serve sentiment_model.h5

# HuggingFace model from Hub
mlship serve bert-base-uncased --source huggingface
```

Same command pattern. Same API format. Same workflow.

## Comparison with Other Tools

| Tool | Frameworks Supported | Code Required | Best For |
|------|---------------------|---------------|----------|
| **mlship** | sklearn, PyTorch, TF, HF | **Zero** | Multi-framework teams, rapid prototyping, learning, benchmarking |
| transformers-serve | HuggingFace only | Zero | HF models exclusively |
| vLLM | LLMs only | Zero | Production LLM serving (high performance) |
| Ollama | LLMs only | Zero | Local LLM chat interfaces |
| BentoML | All frameworks | **Yes** | Production deployments, complex pipelines |
| FastAPI (manual) | All frameworks | **Yes** | Custom requirements, full control |

## mlship's Unique Value

### 1. Multi-Framework Support

mlship is the **only zero-code tool** that works across all major ML frameworks:

- **Traditional ML**: scikit-learn, XGBoost (coming soon)
- **Deep Learning**: PyTorch, TensorFlow
- **Transformers**: HuggingFace models (local and Hub)

Switch frameworks without changing your serving workflow.

### 2. Local-First Philosophy

- **No cloud dependency**: Runs entirely on your machine
- **No API keys**: No rate limits or usage tracking
- **Full control**: Your models and data never leave your system
- **Privacy**: Perfect for sensitive or proprietary models

Compare this to:
- HuggingFace Inference API (requires API token, sends data to HF servers)
- Cloud ML platforms (data leaves your environment)

### 3. Zero Code, True Zero

Unlike other "easy" solutions, mlship requires **absolutely no Python code**:

```python
# BentoML - requires writing service code
import bentoml
from bentoml.io import NumpyNdarray

@bentoml.service(...)
class MyService:
    @bentoml.api
    def predict(self, input: NumpyNdarray):
        return self.model.predict(input)
```

```bash
# mlship - no code, just serve
mlship serve model.pkl
```

### 4. Consistent API

All models, regardless of framework, expose the same REST API:

```bash
# Works for ANY model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

Your application code doesn't need to know which framework you're using.

### 5. Built-in Performance Benchmarking

Measure model serving performance without external tools:

```bash
# Benchmark any model with one command
mlship benchmark model.pkl --requests 100

# Compare frameworks with JSON output
mlship benchmark model.pt --output json > pytorch_results.json
mlship benchmark model.pkl --output json > sklearn_results.json
```

Get cold start latency, avg/min/p50/p95/p99/max latency, and throughput (RPS) — useful for framework comparison, performance regression testing, and CI/CD integration. No other zero-code serving tool includes built-in benchmarking.

### 6. Educational Sweet Spot

Perfect for courses, tutorials, and learning:

- Students learn **model serving concepts** without framework-specific tools
- One tool works for the entire ML curriculum (sklearn → PyTorch → transformers)
- Focus on ML, not deployment infrastructure
- Easy to demo in classrooms or workshops

## Target Audience

### 👨‍🎓 Students & Learners
- Learning multiple ML frameworks
- Want to focus on models, not deployment
- Need quick ways to test models as APIs

### 👩‍💻 Data Scientists
- Prototype models locally before production
- Test models with realistic API interactions
- Share models with teammates without cloud setup

### 👨‍💼 Small Teams
- Work with multiple frameworks
- Don't have dedicated MLOps infrastructure
- Need fast iteration cycles

### 📚 Educators & Authors
- Teach ML model serving
- Create framework-agnostic tutorials
- Need simple, reproducible examples

## Use Cases

### Rapid Prototyping
Train a model → `mlship serve model.pkl` → Test API → Iterate

No time wasted on deployment setup.

### Local Development
Develop applications against real model APIs without cloud dependencies.

### Team Collaboration
Share models with teammates who can serve them with one command, regardless of their ML framework experience.

### Educational Content
Create tutorials that work across frameworks without teaching tool-specific deployment.

### API Testing
Quickly wrap models in APIs to test integration with applications.

## When NOT to Use mlship

mlship is designed for **local development and rapid prototyping**. Consider alternatives for:

- **Production LLM serving at scale** → Use vLLM (optimized for performance)
- **Complex production deployments** → Use BentoML (enterprise features)
- **Interactive demos with UIs** → Use Gradio (built for demos)
- **High-traffic production APIs** → Use framework-specific optimized servers

mlship excels at getting models into API form quickly and easily. For production, you might graduate to specialized tools - but you'll develop and test locally with mlship first.

## Philosophy

### Simplicity Over Features
mlship doesn't try to do everything. It does one thing well: turn ML models into APIs with minimal friction.

### Framework Democracy
No framework is favored. sklearn, PyTorch, TensorFlow, and HuggingFace models are all first-class citizens.

### Local-First, Cloud-Optional
Your development environment shouldn't require internet connectivity or cloud accounts. mlship works entirely offline (except for downloading Hub models).

### Zero Configuration
No YAML files, no config files, no environment setup. Just models and commands.

## Roadmap

mlship continues to evolve while maintaining its core simplicity:

- **More frameworks**: XGBoost, LightGBM support
- **More model sources**: PyTorch Hub, TensorFlow Hub integration
- **Better performance**: Optional batch processing, GPU support
- **Production helpers**: Health checks, monitoring, logging

But the core promise remains: **zero code, any framework, instant API**.

## Get Started

```bash
# Install
pip install mlship

# Serve any model
mlship serve your-model-file

# Or load from HuggingFace Hub
mlship serve bert-base-uncased --source huggingface
```

That's it. No tutorials needed. No configuration required. Just models and commands.

---

**mlship**: The universal ML model server for developers who want to focus on models, not deployment.
