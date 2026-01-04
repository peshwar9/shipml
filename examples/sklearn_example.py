"""
Example: Train a scikit-learn model and serve it with ShipML.

This script:
1. Creates a synthetic dataset
2. Trains a Random Forest classifier
3. Saves the model
4. Shows how to serve it with ShipML

Requirements:
    uv pip install scikit-learn
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import joblib


def main():
    print("🤖 Training scikit-learn model...")
    print()

    # 1. Create synthetic dataset
    print("📊 Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X.shape[1]}")
    print()

    # 2. Train model
    print("🎯 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"   Training accuracy: {train_score:.3f}")
    print(f"   Test accuracy: {test_score:.3f}")
    print()

    # 3. Save model
    model_path = "fraud_detector.pkl"
    print(f"💾 Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print()

    # 4. Instructions
    print("✅ Model saved successfully!")
    print()
    print("🚀 Next steps:")
    print()
    print("   1. Serve your model:")
    print(f"      shipml serve {model_path}")
    print()
    print("   2. Test it:")
    print("      curl -X POST http://localhost:8000/predict \\")
    print('        -H "Content-Type: application/json" \\')
    print(f'        -d \'{{"features": [{", ".join(["1.0"] * 10)}]}}\'')
    print()
    print("   3. View API docs:")
    print("      http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()
