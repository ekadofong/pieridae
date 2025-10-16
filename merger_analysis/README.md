# BYOL + PCA Merger Analysis

Clean, refactored version of the BYOL-based galaxy merger classification pipeline. This package uses self-supervised learning (BYOL) to learn galaxy image representations, followed by PCA-based dimensionality reduction and K-NN label propagation for semi-supervised merger classification.

## Overview

This pipeline:
1. **Trains a BYOL model** on unlabeled galaxy images to learn meaningful representations
2. **Extracts embeddings** from the trained model
3. **Applies PCA and UMAP** for dimensionality reduction and visualization
4. **Propagates labels** using K-NN in PCA space for semi-supervised classification
5. **Identifies merger candidates** based on probability distributions
6. **Analyzes environmental effects** on merger fraction

## Directory Structure

```
merger_analysis/
├── README.md                      # This file
├── config.yaml                    # Configuration file
├── scripts/
│   ├── run_analysis.py           # Main analysis script
│   ├── flag_objects.py           # Flag objects for manual classification
│   └── run_simulated.py          # Test with simulated galaxies
└── notebooks/
    ├── merger_classification.ipynb   # Interactive merger analysis
    └── environment_analysis.ipynb    # Environment-merger relation
```

## Core Module

The refactored core functionality lives in `pieridae.starbursts.byol`:

- **`BYOLModelManager`**: Model setup, training, and embedding extraction with device management (MPS/CUDA/CPU)
- **`EmbeddingAnalyzer`**: PCA and UMAP dimensionality reduction
- **`LabelPropagation`**: K-NN based semi-supervised label propagation
- **`SimulatedGalaxyGenerator`**: Generate test data with known ground truth
- **Utility functions**: Image loading, metric computation, etc.

## Installation

### Requirements

```bash
# Core dependencies
torch>=2.0.0
torchvision
byol-pytorch
numpy
scipy
scikit-learn
umap-learn
pyyaml
tqdm
pandas

# Astronomy packages
astropy

# Custom packages (available in merian environment)
ekfstats      # Starlet wavelet transforms
ekfplot       # Plotting utilities
carpenter     # Emission line corrections
agrias        # Photometry utilities

# Optional
matplotlib
seaborn
jupyter
```

### Setup

```bash
# Activate merian environment
conda activate merian

# Or create new environment
conda create -n merger_analysis python=3.10
conda activate merger_analysis
pip install torch torchvision byol-pytorch umap-learn scikit-learn pyyaml tqdm pandas astropy matplotlib seaborn jupyter
```

## Usage

### 1. Configuration

Edit `config.yaml` to set paths and parameters:

```yaml
data:
  input_path: ../../../local_data/pieridae_output/starlet/msorabove_v0
  output_path: ../../../local_data/byol_results/merger_analysis

training:
  num_epochs: 500
  batch_size: 1024
  learning_rate: 0.0003

labels:
  classifications_file: ../classifications_kadofong_current.csv
  minimum_labeled_neighbors: 5
  n_neighbors: 50
```

### 2. Run Full Pipeline

```bash
cd scripts/

# Full pipeline (training + analysis)
python run_analysis.py --mode full

# Training only
python run_analysis.py --mode train --epochs 500

# Analysis only (requires trained model)
python run_analysis.py --mode analyze

# Custom config
python run_analysis.py --config my_config.yaml --mode full
```

### 3. Flag Objects for Manual Classification

Identify objects with insufficient labeled neighbors:

```bash
python flag_objects.py

# Custom threshold
python flag_objects.py --n-min 8

# Custom output
python flag_objects.py --output-file my_flagged_objects.csv
```

### 4. Test with Simulated Galaxies

Validate the pipeline on simulated data with known ground truth:

```bash
python run_simulated.py

# Custom parameters
python run_simulated.py --n-per-class 500 --epochs 300

# Custom output
python run_simulated.py --output-path ./my_sim_results
```

### 5. Interactive Analysis

Use Jupyter notebooks for interactive exploration:

```bash
cd notebooks/

# Launch Jupyter
jupyter notebook

# Open merger_classification.ipynb
# Or environment_analysis.ipynb
```

## Workflow

### Full Pipeline

```
1. Load Images
   ├─ Read g-band, i-band, high-frequency i-band
   └─ Shape: (N, 3, 150, 150)

2. Train BYOL Model
   ├─ Self-supervised contrastive learning
   ├─ Data augmentations (flips, rotations, color jitter)
   ├─ Checkpointing every 100 epochs
   └─ Device-aware (MPS/CUDA/CPU)

3. Extract Embeddings
   ├─ Forward pass through trained encoder
   └─ Shape: (N, 512)

4. Dimensionality Reduction
   ├─ PCA: (N, 512) → (N, 10)
   │   └─ Automatic component selection (95% variance)
   └─ UMAP: (N, 10) → (N, 2)
       └─ For visualization

5. Label Propagation
   ├─ K-NN in PCA space (k=50)
   ├─ Distance-weighted probability labels
   ├─ Iterative refinement with high-confidence additions
   └─ Flag objects with n_labels < threshold

6. Merger Classification
   ├─ P(merger) = P(ambiguous) + P(merger)
   ├─ Exclude fragmented (P(frag) > 0.3)
   └─ Identify candidates: P(merger) > P(undisturbed)

7. Analysis & Visualization
   ├─ PCA/UMAP scatter plots
   ├─ Confusion matrices
   ├─ Example images
   └─ Environment correlations
```

## Configuration Parameters

### Model Architecture
- `image_size`: Image dimension (default: 150)
- `projection_size`: BYOL projection dimension (default: 128)
- `projection_hidden_size`: Hidden layer size (default: 1024)
- `moving_average_decay`: EMA decay for target network (default: 0.99)

### Training
- `num_epochs`: Training epochs (default: 500)
- `batch_size`: Training batch size (default: 1024)
- `learning_rate`: Adam learning rate (default: 0.0003)
- `save_interval`: Checkpoint frequency (default: 100)
- `resume`: Resume from checkpoint (default: false)

### Analysis
- `pca_components`: Number of PCA components (default: 10, null for auto)
- `explained_variance_threshold`: Variance threshold for auto PCA (default: 0.95)
- `n_neighbors`: K-NN neighbors for UMAP (default: 15)
- `min_dist`: UMAP minimum distance (default: 0.01)

### Label Propagation
- `n_neighbors`: K-NN neighbors for propagation (default: 50)
- `minimum_labeled_neighbors`: Threshold for auto-labels (default: 5)
- `prob_threshold`: Confidence threshold for adding labels (default: 0.6)
- `frag_threshold`: Special threshold for fragmentation class (default: 0.1)

## Output Files

```
output_path/
├── byol_final_model.pt                  # Trained BYOL model
├── model_checkpoint.pt                  # Training checkpoint
├── embeddings.npy                       # Extracted embeddings (N, 512)
├── dimensionality_reduction_results.pkl # PCA/UMAP results
├── merger_analysis_results.pkl          # Full analysis results
├── flagged_for_classification.csv       # Objects needing manual labels
├── embeddings_visualization.png         # PCA/UMAP plots
└── analysis_YYYYMMDD_HHMMSS.log        # Analysis log
```

## Label Meanings

```
0: unclassified   - Not yet manually classified
1: undisturbed    - Normal, non-interacting galaxy
2: ambiguous      - Possible merger, uncertain
3: merger         - Clear merger features
4: fragmentation  - Fragmented/clumpy morphology
5: artifact       - Image artifact or bad data
```

## Device Support

The pipeline automatically detects and uses available hardware:

- **Apple Silicon (MPS)**: Uses Metal Performance Shaders with memory management
- **NVIDIA GPU (CUDA)**: Standard CUDA acceleration
- **CPU**: Fallback with multi-threading

Batch sizes are automatically adjusted for MPS limitations.

## Simulated Galaxy Tests

The `run_simulated.py` script generates three classes of mock galaxies:

1. **Disk**: Single exponential Sersic profile
2. **Double Nuclei**: Two offset Sersic profiles (addition)
3. **Dipole**: Two Sersic profiles with different ellipticities (subtraction)

This provides ground truth for validating classification performance (purity, completeness, confusion matrix).

## Advanced Usage

### Using the Core Module in Your Code

```python
from pieridae.starbursts.byol import (
    BYOLModelManager,
    EmbeddingAnalyzer,
    LabelPropagation,
    load_merian_images
)

# Load data
images, img_names = load_merian_images(data_path)

# Train model
model_manager = BYOLModelManager(config, output_path)
model_manager.train_model(images)

# Extract embeddings
embeddings = model_manager.extract_embeddings(images)

# Dimensionality reduction
analyzer = EmbeddingAnalyzer(config)
embeddings_pca = analyzer.compute_pca(embeddings)

# Label propagation
propagator = LabelPropagation(n_neighbors=50, n_min=5)
propagator.fit_neighbors(embeddings_pca)
prob_labels, n_labels = propagator.estimate_labels(labels)
```

### Custom Augmentations

Modify the `BYOLModelManager.setup_model()` method to use custom data augmentations:

```python
transform_custom = nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.GaussianBlur(kernel_size=3),
    # Add your augmentations here
)
```

### Loading Pre-trained Models

```python
# Load existing model
model_manager = BYOLModelManager(config, output_path)
model_manager.load_trained_model(model_path)

# Extract embeddings without training
embeddings = model_manager.extract_embeddings(new_images)
```

## Troubleshooting

### Out of Memory Errors

```bash
# Reduce batch sizes in config.yaml
training:
  batch_size: 512  # Instead of 1024

inference:
  batch_size: 256  # Instead of 1024
```

### MPS Errors (Apple Silicon)

The code includes automatic CPU fallback for MPS errors. If persistent:

```python
# Force CPU usage
device = torch.device('cpu')
```

### Missing Labels

The pipeline works with partial labels. Objects without labels receive `0` and can be auto-labeled through K-NN propagation.

### Import Errors

Ensure all dependencies are installed:

```bash
pip install torch torchvision byol-pytorch umap-learn scikit-learn
```

For custom packages (ekfstats, ekfplot), use the merian conda environment.

## Citation

If you use this code, please cite:

```bibtex
@article{Kadofong2025,
  title={Self-Supervised Learning for Galaxy Merger Classification},
  author={Kadofong, E. K. and others},
  journal={In Prep},
  year={2025}
}
```

## Original Code

This is a refactored version of the original BYOL analysis code. The original scripts remain in `scripts/byol/`:

- `byol_cluster_analysis_macos.py`
- `flag_for_classification.py`
- `byol_simulated.py`
- `byol_config.yaml`
- `notebooks/BYOL_base_analysis.ipynb`
- `notebooks/MerianEnvironment.ipynb`

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: ekadofong (at) princeton.edu

## License

This code is part of the pieridae package for the Merian survey.
