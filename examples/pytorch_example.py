"""
Example: Train a PyTorch model and serve it with ShipML.

This script:
1. Creates a simple neural network
2. Trains it on synthetic data
3. Saves the full model
4. Shows how to serve it with ShipML

Requirements:
    uv pip install torch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class SimpleClassifier(nn.Module):
    """Simple feedforward neural network."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 classes
        )

    def forward(self, x):
        return self.network(x)


def main():
    print("🤖 Training PyTorch model...")
    print()

    # 1. Create dataset
    print("📊 Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8, n_classes=2, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X.shape[1]}")
    print()

    # 2. Train model
    print("🎯 Training neural network...")
    model = SimpleClassifier(input_dim=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train)
        train_acc = (train_outputs.argmax(1) == y_train).float().mean()

        test_outputs = model(X_test)
        test_acc = (test_outputs.argmax(1) == y_test).float().mean()

    print()
    print(f"   Training accuracy: {train_acc:.3f}")
    print(f"   Test accuracy: {test_acc:.3f}")
    print()

    # 3. Save model (IMPORTANT: Save full model, not just state_dict)
    model_path = "pytorch_classifier.pt"
    print(f"💾 Saving model to {model_path}...")
    torch.save(model, model_path)  # Save full model
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
