# Figure Generation for Merger Analysis

This directory contains scripts for generating publication-quality figures from BYOL merger classification analysis.

## Contents

- **`generate_figures.py`**: Main script for creating galaxy example figures
- **`pull_cutouts.py`**: Utility script for pulling FITS cutouts from tiger3-sumire cluster
- **`figure_data/`**: Directory containing FITS cutouts for figure generation
  - `hsc/`: HSC broadband images (g, r, i, z, y) and PSF files
  - `merian/`: Merian narrowband images (N708, N540)

## Usage

### Generate Galaxy Example Figures

```bash
# Generate figures using default config
python generate_figures.py

# Use custom config
python generate_figures.py --config ../custom_config.yaml

# Choose mass regime (lowmass or highmass)
python generate_figures.py --mass-regime lowmass

# Specify specific galaxy indices
python generate_figures.py --galaxy-indices 100,200,300,400

# Custom output filename
python generate_figures.py --output my_galaxies.pdf
```

### Pull FITS Cutouts

```bash
# Pull cutouts using Merian ID
python pull_cutouts.py M123456

# Pull cutouts using object name
python pull_cutouts.py J095618.67+030835.28

# Specify output directory
python pull_cutouts.py M123456 -o ./my_data/

# Dry run (show commands without executing)
python pull_cutouts.py M123456 --dry-run
```

## Requirements

The figure generation script requires:
- Pre-computed analysis results from `merger_analysis/notebooks/merger_classification.ipynb` or `merger_analysis/scripts/run_analysis.py`
- FITS cutouts in `figure_data/` directory (use `pull_cutouts.py` to obtain them)
- Python packages: numpy, pandas, matplotlib, astropy, pyyaml, tqdm, ekfplot (optional)

## Configuration

The scripts use the configuration file at `../config.yaml` by default. Key settings:

```yaml
data:
  input_path: ../../local_data/pieridae_output/starlet/msorabove_v0
  output_path: ../output/

labels:
  classifications_file: ../../../quick_projects/vizinspect/scripts/classifications_kadofong_current.csv
  n_neighbors: 50
  minimum_labeled_neighbors: 5
  prob_threshold: 0.7
```

## Figure Layout

The generated figures use a transposed grid layout:
- **Rows**: Individual galaxies (typically 4: merger, ambiguous, undisturbed, fragmented)
- **Columns**: Visualization types
  1. r-N708-i RGB composite
  2. HSC i-band (low surface brightness rendering with SymLog normalization)
  3. High-frequency feature reconstruction from starlet decomposition

Each galaxy is annotated with:
- Classification probabilities (Pr[ud], Pr[amb], Pr[merg], Pr[frag])
- Number of labeled neighbors
- Stellar mass (log M*/M☉)

## Algorithms

All classification algorithms and analysis methods are identical to those in:
- `merger_analysis/notebooks/merger_classification.ipynb`
- `merger_analysis/scripts/run_analysis.py`

The figure generation script:
1. Loads pre-computed embeddings, PCA, and UMAP from analysis results
2. Computes K-NN label probabilities using the same parameters
3. Selects representative galaxies based on classification probabilities
4. Creates multi-panel visualization with consistent formatting

## File History

Ported from `pieridae/scripts/figure_generation/` on 2025-10-20:
- `pull_cutouts.py`: Updated import paths for new directory structure
- `galaxy_examples_transposed.ipynb` → `generate_figures.py`: Converted notebook to standalone script with CLI
- `figure_data/`: Copied FITS cutouts directory

## Related Files

- Configuration: `../config.yaml`
- Analysis notebook: `../notebooks/merger_classification.ipynb`
- Analysis script: `../scripts/run_analysis.py`
- Results: `../output/merger_analysis_results.pkl` or `../output/dimensionality_reduction_results.pkl`
