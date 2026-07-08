"""Generates a single synthetic input (flat pixel list) for smoke
testing and benchmarking, reusing model.py's pattern generator."""
import json
import numpy as np
from model import _make_pattern, IMG_SIZE

def make_payload(rng=None):
    """Return a payload dict suitable for the /predict endpoint and the ground‑truth label.
    """
    rng = rng or np.random.default_rng()
    label = int(rng.integers(0, 4))
    img = _make_pattern(label, rng)
    return {"pixels": img.flatten().tolist()}, label

if __name__ == "__main__":
    payload, label = make_payload()
    print(json.dumps(payload))
    print("label", label)
