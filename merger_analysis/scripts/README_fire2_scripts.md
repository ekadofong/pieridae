# FIRE2 Mock Analysis Scripts

This directory contains scripts for analyzing FIRE2 simulation mock images using semi-supervised BYOL classification.

## Scripts Overview

### 1. `run_fire2_mock.py` - Full Pipeline

**Purpose**: Complete pipeline from image loading to final classification.

**What it does**:
1. Loads FIRE2 mock images from `../local_data/mock_images/`
2. Preprocesses images (creates 3-channel g/i/HF images)
3. **Trains BYOL model** from scratch (~10min on CPU, ~2min on GPU)
4. Extracts embeddings and applies PCA
5. Creates sparse training labels (semi-supervised setup)
6. Runs K-NN label propagation
7. Computes metrics and generates visualizations

**When to use**:
- First time analysis with new images
- Testing different BYOL architectures
- Changing number of epochs or batch size
- Need to regenerate embeddings

**Example usage**:
```bash
# Basic usage with 3 galaxies, 1000 images each, 30% training
python run_fire2_mock.py --tags m11h_res7100 m11d_res7100 m11e_res7100 \
    --n-per-galaxy 1000 --train-fraction 0.3 --epochs 100

# Quick test with fewer epochs
python run_fire2_mock.py --tags m11h_res7100 m11d_res7100 m11e_res7100 \
    --n-per-galaxy 500 --epochs 50
```

**Output**: `../output/fire2_mock/`
- `byol_final_model.pt` - Trained BYOL model
- `embeddings.npy` - BYOL embeddings
- `dimensionality_reduction_results.pkl` - PCA results
- `probability_labels.npy` - Final predictions
- Metrics and visualizations

---

### 2. `run_fire2_problabels.py` - Fast PCA & Label Propagation Recomputation

**Purpose**: Quickly experiment with **PCA dimensionality and label propagation** parameters **without retraining BYOL**.

**What it does**:
1. **Loads pre-computed** BYOL embeddings from `run_fire2_mock.py` output
2. **Recomputes PCA** with configurable dimensionality (n_components, variance threshold)
3. Optionally creates new train/test split or reuses existing
4. Runs K-NN label propagation with configurable parameters
5. Computes metrics and generates visualizations

**When to use**:
- Testing different PCA dimensionalities (curse of dimensionality experiments)
- Experimenting with different K values for KNN
- Testing confidence thresholds
- Trying different train/test splits
- Rapid iteration on both PCA and classification parameters

**Speed**: ~0.1 seconds for PCA + ~0.01 seconds for KNN (vs ~10 minutes for full pipeline)

**Example usage**:
```bash
# Recompute with different PCA dimensionality
python run_fire2_problabels.py --pca-components 5
python run_fire2_problabels.py --pca-components 20
python run_fire2_problabels.py --variance-threshold 0.99

# Experiment with both PCA and KNN
python run_fire2_problabels.py --pca-components 10 --n-neighbors 100

# Use pre-computed PCA, only change KNN (faster)
python run_fire2_problabels.py --use-precomputed-pca --n-neighbors 100

# Lower confidence threshold to get more pseudo-labels
python run_fire2_problabels.py --use-existing-split \
    --pca-components 15 --prob-threshold 0.8 --n-min-auto 10

# New train/test split with different fraction
python run_fire2_problabels.py --train-fraction 0.1 --random-seed 123

# Test dimensionality sweep
for n in 5 10 20 50; do
    python run_fire2_problabels.py --pca-components $n \
        --output-dir ../output/pca_sweep/pca${n}comp
done
```

**Output**: `../output/fire2_mock/problabels_recompute/` (or custom directory)
- `probability_labels.npy` - Recomputed predictions
- `recomputed_pca.pkl` - New PCA model and embeddings (if PCA was recomputed)
- `label_propagation_results.pkl` - Propagation details
- Updated metrics and visualizations

---

## Relationship Between Scripts

```
run_fire2_mock.py          run_fire2_problabels.py
     (slow)                    (medium-fast)
       │                            │
       ├─ Load images               │
       ├─ Train BYOL (~10min)       │
       ├─ Extract embeddings ───────┼─ Load BYOL embeddings
       ├─ Compute PCA           ────┼─ Recompute PCA (NEW params, ~0.1s)
       ├─ Label propagation         ├─ Label propagation (NEW params, ~0.01s)
       ├─ Metrics & viz             └─ Metrics & viz
       └─ Save results
```

**Analogy**: Similar to `run_analysis.py` vs `merger_classification.ipynb`:
- `run_fire2_mock.py` ≈ Full analysis notebook (do once, expensive)
- `run_fire2_problabels.py` ≈ Quick parameter tuning (iterate many times, cheap)

---

## Parameters Reference

### PCA Parameters (run_fire2_problabels.py only)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pca-components` | auto | Number of PCA components (fixed dimensionality) |
| `--variance-threshold` | 0.95 | Explained variance for auto component selection |
| `--use-precomputed-pca` | False | Skip PCA recomputation, use existing |

**Effect of PCA dimensionality**:
- **Too few components** (e.g., 5): May lose discriminative information
- **Too many components** (e.g., 50): Curse of dimensionality, KNN performance degrades
- **Optimal range**: Typically 10-20 components for 3-class problems
- **Variance threshold**: 0.95 = capture 95% of variance (conservative), 0.99 = 99% (aggressive)

### Label Propagation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-neighbors` | 50 | Number of nearest neighbors to consider |
| `--n-min` | 5 | Minimum labeled neighbors to make prediction |
| `--n-min-auto` | 15 | Minimum neighbors to add pseudo-label |
| `--prob-threshold` | 0.9 | Confidence threshold for pseudo-labeling |

**Effect of increasing K (`--n-neighbors`)**:
- ✅ More robust to local noise
- ✅ Better pseudo-label coverage
- ⚠️ May blur class boundaries
- ⚠️ Interacts with PCA dimensionality (higher dim → need higher K)
- Results: K=50: 80% accuracy, K=100: 95% accuracy

**Effect of lowering threshold (`--prob-threshold`)**:
- ✅ More pseudo-labels added
- ⚠️ Risk of propagating errors
- threshold=0.9: conservative, threshold=0.7: aggressive

### Train/Test Split Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train-fraction` | 0.3 | Fraction of data for training |
| `--random-seed` | 42 | Random seed for reproducibility |
| `--no-stratified-split` | False | Disable stratified sampling |
| `--use-existing-split` | False | Reuse previous split |

---

## Typical Workflow

### Initial Run
```bash
# 1. Generate embeddings (do once, ~10min)
python run_fire2_mock.py --tags m11h_res7100 m11d_res7100 m11e_res7100 \
    --n-per-galaxy 1000 --train-fraction 0.3

# Check results in ../output/fire2_mock/
```

### Experimentation
```bash
# 2. Experiment with PCA dimensionality (each takes ~0.1sec)

# Try different PCA components
python run_fire2_problabels.py --pca-components 5 --output-dir ../output/pca5
python run_fire2_problabels.py --pca-components 10 --output-dir ../output/pca10
python run_fire2_problabels.py --pca-components 20 --output-dir ../output/pca20

# Try different variance thresholds
python run_fire2_problabels.py --variance-threshold 0.99

# 3. Experiment with KNN parameters (each takes ~0.01sec with precomputed PCA)

# Try larger K (use precomputed PCA for speed)
python run_fire2_problabels.py --use-precomputed-pca --n-neighbors 100

# Try lower threshold
python run_fire2_problabels.py --use-precomputed-pca --prob-threshold 0.7

# 4. Joint optimization (PCA + KNN)
python run_fire2_problabels.py --pca-components 15 --n-neighbors 100 \
    --output-dir ../output/pca15_k100
```

### Comparison
```bash
# 3. Compare multiple runs
ls -lt ../output/fire2_mock/problabels_*/classification_metrics_hard.json
```

---

## Output Files

### Common to both scripts:
- `classification_metrics_hard.json` - Confusion matrix, precision/recall
- `probabilistic_metrics.json` - Confidence, entropy, top-k accuracy
- `probability_labels.npy` - Probability distributions (N x n_classes)
- `evaluation_results.png` - PCA plots and confusion matrix
- `probability_analysis.png` - Confidence distributions

### Only in `run_fire2_mock.py`:
- `byol_final_model.pt` - Trained model weights
- `embeddings.npy` - Full BYOL embeddings
- `dimensionality_reduction_results.pkl` - PCA model + embeddings
- `fire2_mock_data.pkl` - Original images and metadata
- `sample_fire2_galaxies.png` - Example images
- `qa_figure.png` - Broadband vs HF comparison

### Only in `run_fire2_problabels.py`:
- `label_propagation_results.pkl` - Detailed propagation statistics

---

## Troubleshooting

### Error: "Data file not found"
**Solution**: Run `run_fire2_mock.py` first to generate embeddings.

### Error: "No images found for galaxy X"
**Solution**: Generate mock images first:
```bash
cd ../local_data/
python generate_fire2_images.py --tag m11h_res7100 --n-images 1000
```

### Low accuracy despite balanced data
**Possible causes**:
1. Too few training samples → increase `--train-fraction`
2. Too small K → increase `--n-neighbors`
3. Classes not separable → check PCA plots in `evaluation_results.png`

### All predictions collapse to one class
**Fixed in latest version!** This was caused by 0-indexed labels being treated as unlabeled. If you still see this:
1. Make sure you're using the latest `run_fire2_mock.py`
2. Check that `sparse_labels` uses 1-indexed labels (see commit fixing label indexing)

---

## Performance Benchmarks

| Task | Time (CPU) | Time (GPU) |
|------|------------|------------|
| Full pipeline (1000 imgs/galaxy, 100 epochs) | ~10 min | ~2 min |
| Recompute prob labels | ~1 sec | ~1 sec |
| Load and visualize only | ~0.5 sec | ~0.5 sec |

**Speedup ratio**: ~600x faster to iterate with `run_fire2_problabels.py`

---

## Citation

If using these scripts, please cite:
- BYOL: Grill et al. (2020) "Bootstrap your own latent"
- FIRE2 simulations: Hopkins et al. (2018)
