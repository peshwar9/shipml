"""
Example: Download a Hugging Face model and serve it with ShipML.

This script:
1. Downloads a pre-trained sentiment analysis model from Hugging Face
2. Saves it locally
3. Shows how to serve it with ShipML

Requirements:
    uv pip install transformers torch
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    print("🤗 Downloading Hugging Face model...")
    print()

    # Model to use (small and fast for demo)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    save_path = "./sentiment-model"

    print(f"📦 Model: {model_name}")
    print(f"💾 Saving to: {save_path}")
    print()

    # Download model and tokenizer
    print("⬇️  Downloading model (this may take a minute)...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save locally
    print("💾 Saving model locally...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print()
    print("✅ Model downloaded and saved successfully!")
    print()
    print("📊 Model Information:")
    print(f"   Task: Sentiment Analysis (Positive/Negative)")
    print(f"   Model: DistilBERT (distilled BERT)")
    print(f"   Size: ~250 MB")
    print()

    # Instructions
    print("🚀 Next steps:")
    print()
    print("   1. Serve your model:")
    print(f"      shipml serve {save_path}")
    print()
    print("   2. Test it:")
    print("      curl -X POST http://localhost:8000/predict \\")
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"features": "This product is amazing!"}\'')
    print()
    print("   3. Try different inputs:")
    print('      {"features": "I love this!"}  # Positive')
    print('      {"features": "This is terrible."}  # Negative')
    print('      {"features": "It\'s okay, nothing special."}  # Neutral-ish')
    print()
    print("   4. Batch prediction:")
    print('      {"features": ["Great!", "Bad!", "Meh."]}')
    print()
    print("   5. View API docs:")
    print("      http://localhost:8000/docs")
    print()
    print("💡 Tip: The first prediction may be slow (~2 seconds) as the model")
    print("   initializes. Subsequent predictions are much faster (<100ms).")
    print()


if __name__ == "__main__":
    main()
