"""
model.py

Defines a small CNN classifier and utilities to generate a synthetic
image-classification dataset and train the model on it.

Synthetic task: 1-channel 28x28 images belonging to one of 4 classes.
Each class is generated from a distinct geometric pattern (horizontal
stripes, vertical stripes, diagonal gradient, checkerboard) plus noise.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 28
NUM_CLASSES = 4
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_weights.pt")


class TinyCNN(nn.Module):
    """A small CNN, deliberately lightweight so CPU inference is fast
    enough to demonstrate real throughput differences under load."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _make_pattern(label: int, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    if label == 0:  # horizontal stripes
        for r in range(IMG_SIZE):
            if (r // 3) % 2 == 0:
                img[r, :] = 1.0
    elif label == 1:  # vertical stripes
        for c in range(IMG_SIZE):
            if (c // 3) % 2 == 0:
                img[:, c] = 1.0
    elif label == 2:  # diagonal gradient
        for r in range(IMG_SIZE):
            for c in range(IMG_SIZE):
                img[r, c] = ((r + c) % IMG_SIZE) / IMG_SIZE
    else:  # checkerboard
        for r in range(IMG_SIZE):
            for c in range(IMG_SIZE):
                if (r // 4 + c // 4) % 2 == 0:
                    img[r, c] = 1.0

    noise = rng.normal(0, 0.15, size=img.shape).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)
    return img


def generate_dataset(n_per_class: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for label in range(NUM_CLASSES):
        for _ in range(n_per_class):
            images.append(_make_pattern(label, rng))
            labels.append(label)
    images = np.stack(images)[:, None, :, :]  # (N, 1, 28, 28)
    labels = np.array(labels, dtype=np.int64)

    perm = rng.permutation(len(labels))
    return images[perm], labels[perm]


def train_and_save(epochs: int = 8, batch_size: int = 32, lr: float = 1e-3):
    images, labels = generate_dataset()
    split = int(0.85 * len(labels))
    x_train = torch.tensor(images[:split])
    y_train = torch.tensor(labels[:split])
    x_val = torch.tensor(images[split:])
    y_val = torch.tensor(labels[split:])

    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = x_train.size(0)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            val_out = model(x_val)
            val_acc = (val_out.argmax(dim=1) == y_val).float().mean().item()
        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"train_loss={total_loss / n:.4f}  val_acc={val_acc:.3f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved trained weights to {MODEL_PATH}")
    return model


def load_model() -> TinyCNN:
    model = TinyCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    train_and_save()
