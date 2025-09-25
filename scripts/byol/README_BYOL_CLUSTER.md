# BYOL Analysis on Tiger Cluster

This directory contains scripts for running BYOL (Bootstrap Your Own Latent) analysis on Princeton's Tiger cluster, converted from the original Jupyter notebook `byol_apply_tweaked.ipynb`.

## Files Overview

All BYOL scripts are located in `scripts/byol/`:

- **`byol_cluster_analysis.py`** - Main Python script containing the BYOL analysis pipeline
- **`byol_config.yaml`** - Configuration file with all parameters
- **`submit_byol_train.slurm`** - Slurm script for GPU training phase
- **`submit_byol_analyze.slurm`** - Slurm script for CPU analysis phase
- **`submit_byol_full.slurm`** - Slurm script for complete pipeline
- **`run_byol.py`** - Utility script for job management and monitoring

## Quick Start

### 1. Setup Environment
```bash
# Ensure you have the required conda environment
conda activate merenv  # or your environment name

# Verify required packages are installed:
# torch, torchvision, byol-pytorch, numpy, pandas, scikit-learn,
# umap-learn, matplotlib, tqdm, pyyaml
```

### 2. Configure Analysis
Edit `byol_config.yaml` to adjust parameters:
- Data paths
- Model hyperparameters
- Training settings
- Analysis parameters

### 3. Submit Jobs

#### Option A: Full Pipeline (Recommended)
```bash
# From the scripts/byol directory
cd scripts/byol
python run_byol.py submit --mode full
```

#### Option B: Separate Jobs
```bash
# From the scripts/byol directory
cd scripts/byol

# Submit training job (GPU)
python run_byol.py submit --mode train

# After training completes, submit analysis job (CPU)
python run_byol.py submit --mode analyze
```

#### Option C: Direct Slurm Submission
```bash
# From project root directory
sbatch scripts/byol/submit_byol_full.slurm

# Or individual components
sbatch scripts/byol/submit_byol_train.slurm
sbatch scripts/byol/submit_byol_analyze.slurm
```

### 4. Monitor Jobs
```bash
# From the scripts/byol directory
cd scripts/byol

# Check job status
python run_byol.py status

# Monitor specific job
python run_byol.py monitor --job-id <JOB_ID>

# View job logs
python run_byol.py logs --job-id <JOB_ID>
```

## Pipeline Stages

### 1. Data Loading
- Loads image data from pickle files in `local_data/pieridae_output/starlet/starbursts_v0/`
- Supports multi-band images (g, i bands + high-frequency component)
- Automatically handles missing files

### 2. BYOL Training (GPU Phase)
- Uses ResNet18 backbone with ImageNet pretrained weights
- Self-supervised learning with data augmentations
- Automatic checkpointing every 10 epochs
- Resumable from checkpoints
- **Resource Requirements**: 1 GPU, 8 CPUs, 64GB RAM, ~4 hours

### 3. Embedding Extraction
- Extracts learned representations from trained model
- Batch processing for memory efficiency
- Saves embeddings as `.npy` files

### 4. Dimensionality Reduction & Analysis (CPU Phase)
- **PCA**: Reduces embedding dimensionality
- **UMAP**: Creates 2D visualization embeddings
- **Similarity Analysis**: Computes cosine similarity matrix
- **Visualization**: Generates plots and summaries
- **Resource Requirements**: 16 CPUs, 128GB RAM, ~2 hours

## Output Files

Results are saved to `byol_results/` with timestamp subdirectories:

### Training Output
- `byol_final_model.pt` - Trained BYOL model
- `model_checkpoint.pt` - Latest training checkpoint
- `embeddings.npy` - Extracted embeddings

### Analysis Output
- `dimensionality_reduction_results.pkl` - PCA/UMAP results
- `similarity_analysis.pkl` - Similarity matrix and top pairs
- `pca_analysis.png` - PCA component analysis plot
- `embeddings_comparison.png` - PCA vs UMAP visualization
- `analysis_summary.json` - Summary statistics
- `byol_analysis.log` - Detailed logs

## Configuration Options

### Model Parameters
- `image_size`: Input image dimensions
- `projection_size`: Final embedding size
- `projection_hidden_size`: Hidden layer size
- `moving_average_decay`: EMA coefficient

### Training Parameters
- `num_epochs`: Training duration
- `batch_size`: Training batch size
- `learning_rate`: Optimizer learning rate
- `save_interval`: Checkpoint frequency

### Analysis Parameters
- `pca_components`: Number of PCA components
- `umap_components`: UMAP output dimensions
- `n_neighbors`: UMAP neighborhood size
- `min_dist`: UMAP cluster tightness

## Resource Usage Guidelines

### GPU Jobs (Training)
- Use `partition=gpu`
- Request 1 GPU (`gres=gpu:1`)
- 8-16 CPUs typically sufficient
- 32-64GB RAM depending on batch size
- Time limit: 2-6 hours depending on epochs

### CPU Jobs (Analysis)
- Use `partition=cpu`
- 16-32 CPUs for PCA/UMAP parallelization
- 64-128GB RAM for large similarity matrices
- Time limit: 1-3 hours

## Advanced Usage

### Custom Configuration
```bash
# Use custom config file (from scripts/byol directory)
cd scripts/byol
python byol_cluster_analysis.py --config my_config.yaml --mode full
```

### Command Line Overrides
```bash
# Override specific parameters (from scripts/byol directory)
cd scripts/byol
python byol_cluster_analysis.py \
    --mode train \
    --epochs 100 \
    --batch-size 16 \
    --data-path /scratch/gpfs/$USER/my_data
```

### Resuming Training
```bash
# Training automatically resumes from checkpoints when --resume flag is used
cd scripts/byol
python byol_cluster_analysis.py --mode train --resume
```

### Memory Optimization
For large datasets, consider:
- Reducing batch sizes
- Using `low_memory_mode` in config
- Staging data to `/scratch/gpfs` for better I/O

## Troubleshooting

### Common Issues

1. **Out of Memory (GPU)**
   - Reduce `training.batch_size` in config
   - Use smaller model (`projection_hidden_size`)

2. **Out of Memory (CPU)**
   - Reduce `inference.batch_size`
   - Use fewer PCA components
   - Enable `low_memory_mode`

3. **Data Not Found**
   - Verify `data.input_path` in config
   - Check file permissions
   - Ensure data is accessible from compute nodes

4. **Job Fails to Start**
   - Check account/partition settings
   - Verify conda environment exists
   - Check Slurm resource limits

### Debug Mode
```bash
# Run with verbose logging (from scripts/byol directory)
cd scripts/byol
python byol_cluster_analysis.py --mode full --output-path debug_results

# Check logs
tail -f debug_results/byol_analysis.log
```

### Performance Monitoring
```bash
# Monitor GPU usage during training
nvidia-smi -l 1

# Monitor CPU/memory during analysis
htop
```

## Support

For Tiger cluster specific issues:
- Check Princeton Research Computing documentation
- Contact: cses@princeton.edu

For BYOL analysis issues:
- Review logs in output directory
- Check configuration parameters
- Verify data format compatibility