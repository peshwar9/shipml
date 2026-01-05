# ShipML Architecture & Design Decisions

This document explains ShipML's architecture, design philosophy, and key technical decisions.

## Design Philosophy

**ShipML is "demo-ready," not production-ready.**

The core principle is **simplicity over optimization**. ShipML is designed for:
- Students demoing ML projects
- Data scientists prototyping locally
- Educators teaching API concepts
- Quick model validation

It is **not** designed for:
- Production deployments at scale
- High-throughput serving
- Multi-model orchestration
- Advanced optimization scenarios

## Performance & Optimization

### What ShipML Already Optimizes

ShipML includes sensible defaults and optimizations that work out-of-the-box:

1. **Batch Processing**
   - All loaders support batch inputs (`List[List[float]]` or `List[str]`)
   - Framework-native batching (sklearn arrays, PyTorch tensors, etc.)
   - No custom batching logic needed

2. **Model Eval Mode**
   - PyTorch models loaded with `model.eval()` (disables dropout, batchnorm)
   - Reduces memory and improves inference speed
   - Standard practice for inference

3. **CPU Optimization**
   - HuggingFace models use `device=-1` (CPU-optimized)
   - No GPU dependencies required
   - Works on any machine out-of-the-box

4. **Framework Defaults**
   - Leverages sklearn's optimized predict()
   - PyTorch's torch.no_grad() context
   - TensorFlow's eager execution optimizations
   - HuggingFace pipeline's built-in KV caching

### What ShipML Explicitly Doesn't Do

We **intentionally avoid** advanced optimizations to maintain simplicity:

#### ❌ Quantization (Model Compression)

**What it is:** Reducing model precision (FP32 → INT8/FP16) for smaller models and faster inference.

**Why we don't include it:**
- Adds significant complexity (framework-specific implementations)
- Requires careful accuracy vs speed trade-offs
- Not all models support quantization easily
- Users can pre-quantize models before serving if needed

**When to use it:** Production deployments with strict latency/memory requirements.

**Tools to use instead:** PyTorch quantization, TensorFlow Lite, ONNX Runtime

#### ❌ KV Cache Optimization

**What it is:** Caching key-value pairs in transformer models for faster sequential generation.

**Why we don't include it:**
- Already handled by HuggingFace `transformers.pipeline()` by default
- Only critical for long conversational sequences (chatbots, streaming)
- ShipML focuses on single-shot predictions, not stateful conversations
- Adds complexity for minimal benefit in demo scenarios

**When to use it:** Production LLM serving with streaming or chat

**Tools to use instead:** vLLM, Text Generation Inference, OpenLLM

#### ❌ GPU Acceleration (Currently)

**Status:** On roadmap but not implemented

**Why not yet:**
- Adds installation complexity (CUDA, drivers, etc.)
- Breaks "works offline anywhere" guarantee
- Most demo/testing scenarios run fine on CPU
- Would require per-framework GPU handling

**When to add it:** As optional flag (`--device gpu`) without breaking CPU-only workflows

#### ❌ Model Caching/Pooling

**What it is:** Keeping models in memory across requests, connection pooling, etc.

**Why we don't include it:**
- ShipML uses uvicorn which handles this at the ASGI level
- Model is loaded once at startup, not per request
- Simple single-process model is easier to reason about
- Advanced caching (Redis, etc.) is production concern

#### ❌ Advanced Batching Strategies

**What it is:** Dynamic batching, request queuing, adaptive batch sizing

**Why we don't include it:**
- Adds significant complexity to request handling
- Most demo use cases have low concurrent traffic
- Frameworks already batch at the numpy/tensor level
- Production tools (TorchServe, Triton) do this better

**When to use it:** High-throughput production serving

**Tools to use instead:** TorchServe, TensorFlow Serving, Triton Inference Server

### When to Graduate to Production Tools

If you need any of these, consider using specialized serving tools:

| Need | Tool Recommendation |
|------|-------------------|
| GPU inference + batching | TorchServe, TensorFlow Serving |
| LLM serving with streaming | vLLM, Text Generation Inference |
| Multi-model orchestration | Seldon, KServe |
| Auto-scaling | Kubernetes + KServe |
| Quantization + ONNX | ONNX Runtime, TensorRT |
| Multi-framework serving | Triton Inference Server |

## Architecture Overview

### Component Structure

```
shipml/
├── cli.py              # CLI entry point (Click)
├── server.py           # FastAPI app generator
├── loaders/            # Framework-specific loaders
│   ├── base.py         # Abstract base class
│   ├── sklearn.py      # Scikit-learn loader
│   ├── pytorch.py      # PyTorch loader
│   ├── tensorflow.py   # TensorFlow/Keras loader
│   └── huggingface.py  # HuggingFace Transformers loader
├── pipeline.py         # Custom pre/post processing
├── models.py           # Pydantic request/response models
├── errors.py           # Custom exceptions
└── utils.py            # Helper utilities
```

### Request Flow

1. **CLI** (`shipml serve model.pkl`)
   - Detects framework from file extension
   - Loads model using appropriate loader
   - Creates FastAPI app with model embedded
   - Starts uvicorn server

2. **Request** (`POST /predict`)
   - FastAPI validates request (Pydantic)
   - Optional: Custom pipeline preprocesses input
   - Loader validates input format
   - Model prediction runs
   - Optional: Custom pipeline postprocesses output
   - FastAPI serializes response

3. **Model Loading** (One-time at startup)
   - Framework detection (file extension + content sniffing)
   - Loader-specific loading (joblib, torch.load, tf.keras.load_model, etc.)
   - Metadata extraction (input features, model type, etc.)
   - Model kept in memory for duration of server

### Design Patterns

#### 1. Strategy Pattern (Loaders)

Each framework has its own loader implementing the `ModelLoader` interface:
- `load()` - Framework-specific model loading
- `predict()` - Framework-specific prediction
- `validate_input()` - Framework-specific validation
- `get_metadata()` - Extract model information

**Why:** Different frameworks have completely different APIs and conventions

#### 2. Decorator Pattern (Pipelines)

Custom pipelines wrap model prediction with pre/post processing:
- `preprocess()` - Transform API request → model input
- `postprocess()` - Transform model output → API response

**Why:** Allows users to add custom logic without modifying ShipML code

#### 3. Factory Pattern (Server Creation)

`create_app(model, loader, name, pipeline)` dynamically generates FastAPI apps:
- Different request models based on framework (text vs numeric)
- Different validation logic
- Different response formats

**Why:** One server template works for all frameworks

## Key Technical Decisions

### 1. Why FastAPI?

**Alternatives considered:** Flask, Django REST Framework, raw ASGI

**Why FastAPI:**
- Automatic OpenAPI docs (`/docs` endpoint)
- Pydantic validation (type safety + validation)
- Async support (future-proof for async frameworks)
- Modern Python (type hints, async/await)
- Great developer experience

### 2. Why No Database?

**Decision:** ShipML is stateless - no DB, no caching layer, no session storage

**Rationale:**
- Adds complexity and dependencies
- Demo scenarios don't need persistence
- Models are loaded from file system
- Keeps "one command" deployment promise

### 3. Why CPU-Only by Default?

**Decision:** Default to CPU, with GPU on roadmap as optional

**Rationale:**
- Works on any machine (no CUDA required)
- Installation is simple (`pip install shipml`)
- Most demo/testing scenarios don't need GPU
- Avoids GPU driver/version hell
- Maintains "works offline" guarantee

### 4. Why Dynamic Request Models?

**Decision:** Request schema changes based on framework and pipeline

**Rationale:**
- HuggingFace needs text (`{"features": "text"}`)
- Sklearn needs arrays (`{"features": [1.0, 2.0]}`)
- Custom pipelines need flexible JSON
- FastAPI generates correct OpenAPI spec automatically

### 5. Why No Docker (Yet)?

**Decision:** Native Python deployment only, Docker on roadmap

**Rationale:**
- "One command" is simpler than Docker setup
- Students/educators may not know Docker
- Reduces dependencies and installation complexity
- Docker deployment is a future feature, not replacement

## Future Considerations

### Potential Features (Aligned with Philosophy)

✅ **These fit ShipML's mission:**
- GPU support (optional flag: `--device gpu`)
- Batch prediction endpoint (`/predict/batch`)
- Model versioning (serve multiple versions)
- Basic metrics endpoint (`/metrics`)
- Docker deployment option (`shipml docker model.pkl`)

❌ **These don't fit (too complex):**
- Multi-model serving
- A/B testing framework
- Model monitoring/observability
- Auto-scaling
- Distributed inference

### Roadmap Priorities

Based on user feedback and philosophy alignment:

1. **GPU inference** - Valuable but must stay optional
2. **Batch prediction optimization** - Natural extension of current batching
3. **Docker deployment** - Package as optional convenience
4. **XGBoost/LightGBM support** - Commonly requested frameworks
5. **Model metrics endpoint** - Simple addition (`/metrics` for prometheus)

Not prioritized:
- Advanced optimization (quantization, pruning) - Production concern
- Multi-model serving - Adds too much complexity
- Cloud deployment - Out of scope (use cloud-native tools)

## Contributing Guidelines

When adding features, ask:
1. **Does it add significant complexity?** If yes, reconsider
2. **Can it be optional?** Prefer flags over always-on features
3. **Does it break "one command"?** If yes, make it a separate command
4. **Is it demo-ready or production-ready?** Focus on demo-ready

Examples:
- ✅ `--pipeline` flag - Optional, doesn't break simple use case
- ✅ GPU support as `--device gpu` - Optional enhancement
- ❌ Requiring config file - Breaks "one command" promise
- ❌ Built-in monitoring - Adds complexity, production concern

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [HuggingFace Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [PyTorch Inference Mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html)
