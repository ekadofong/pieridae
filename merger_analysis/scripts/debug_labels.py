#!/usr/bin/env python3
"""Debug script to check label matching"""

import sys
from pathlib import Path
import yaml
import pandas as pd

# Add pieridae to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts.byol import load_merian_images

# Load config
with open('../configs/galaxyzoo.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load images
print("Loading images...")
images, img_names = load_merian_images(
    Path(config['data']['input_path']),
    None
)

print(f"\nLoaded {len(img_names)} images")
print(f"First 10 image names: {img_names[:10]}")

# Load labels
label_file = Path(config['labels']['classifications_file'])
print(f"\nLoading labels from: {label_file}")
mergers = pd.read_csv(label_file, index_col=0)

print(f"Total labels in CSV: {len(mergers)}")
print(f"Labels > 0: {(mergers['classification'] > 0).sum()}")

# Reindex to match images
labels = mergers.reindex(img_names)
labels_values = labels.replace(pd.NA, 0).values.flatten().astype(int)

print(f"\nMatched labels: {len(labels_values)}")
print(f"Matched labels > 0: {(labels_values > 0).sum()}")

import numpy as np
unique, counts = np.unique(labels_values, return_counts=True)
print("\nLabel distribution:")
for val, count in zip(unique, counts):
    print(f"  {val}: {count}")
