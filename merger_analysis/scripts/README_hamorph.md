# H-alpha Morphology Analysis Scripts

Scripts for measuring H-alpha and z-band morphology for Merian galaxies using `statmorph_joint`.

## Overview

The H-alpha morphology analysis pipeline consists of:

1. **`calc_hamorph.py`** - Main Python script that performs morphology measurements
2. **`run_hamorph.sh`** - Bash wrapper script for easy execution (local or cluster)
3. **`submit_hamorph.slurm`** - SLURM batch file for parallel processing on Princeton cluster
4. **`hamorph_config.yaml`** - Configuration file with all analysis parameters

## Quick Start

### Local Testing (Serial Mode)

Process a small subset of galaxies locally:

```bash
cd merger_analysis/scripts

# Run with default config (only processes galaxies with available cutouts)
./run_hamorph.sh

# Verbose output
./run_hamorph.sh --verbose

# Use different H-alpha method
./run_hamorph.sh --ha-method zscale
```

### Cluster Execution (SLURM Job Array)

For processing the full catalog in parallel on the Princeton cluster:

```bash
cd merger_analysis/scripts

# Submit all 50 chunks (processes all galaxies with cutouts)
sbatch --array=0-49 submit_hamorph.slurm

# Submit subset for testing (first 5 chunks)
sbatch --array=0-4 submit_hamorph.slurm

# Submit single chunk for debugging
sbatch --array=0 submit_hamorph.slurm

# With custom config file
sbatch --array=0-49 submit_hamorph.slurm /path/to/custom_config.yaml
```

## Analysis Methods

Three methods are available for H-alpha image generation (set via `ha_im_method` in config):

1. **`plaw_pixbypix`** (default) - Pixel-by-pixel powerlaw continuum fit
   - Most accurate for spatially varying continuum
   - Slower but recommended for science-grade results

2. **`zscale`** - Global z-band scaling
   - Fast, simple continuum subtraction
   - Uses single scaling factor across entire image

3. **`ri_avg`** - Average of r and i bands
   - Alternative continuum estimate
   - Useful for comparison

## File Structure

```
merger_analysis/
├── configs/
│   └── hamorph_config.yaml          # Configuration file
├── scripts/
│   ├── calc_hamorph.py              # Main analysis script
│   ├── run_hamorph.sh               # Wrapper script
│   ├── submit_hamorph.slurm         # SLURM batch file
│   └── README_hamorph.md            # This file
└── local_data/
    └── hamorph/                      # Output directory (created automatically)
        └── M{objectid}_morph.npy     # Morphology results (one per galaxy)
```

## Configuration

Edit `../configs/hamorph_config.yaml` to customize:

### Key Parameters

```yaml
# Catalog settings
catalog:
  catalog_file: /path/to/catalog.parquet
  mask_name: is_good                  # Which catalog mask to use

# Cutout paths
cutout_data:
  image_path: /path/to/cutouts/       # HSC and Merian cutouts

# Analysis
analysis:
  ha_im_method: plaw_pixbypix         # zscale | plaw_pixbypix | ri_avg
  plawbands: riz                      # Bands for powerlaw fit
  post_smooth: false                  # Post-smoothing (plaw_pixbypix only)

# Output
output:
  save_dir: /path/to/output/
  save_results: true

# Processing
processing:
  n_chunks: 50                        # Number of SLURM array chunks
  skip_objects: []                    # Object IDs to skip
```

### Important Notes

- **Automatic filtering**: By default, the script only processes galaxies that have cutouts available in `image_path`. Use `--skip-cutout-check` to disable this.
- **Chunk size**: The `n_chunks` parameter should match the `--array` range in your SLURM submission (e.g., `--array=0-49` for 50 chunks).
- **Cutout structure**: Expects subdirectories `hsc/` and `merian/` with standard naming conventions.

## Output Format

Each galaxy produces one `.npy` file: `{objectid}_morph.npy`

The file contains a tuple with three elements:
1. **`morph_z`** - z-band morphology measurements
2. **`morph_joint`** - Joint z-band + H-alpha morphology measurements
3. **`morph_ha`** - H-alpha-only morphology measurements

These are `statmorph_joint` objects with properties like:
- `gini`, `m20`, `concentration`, `asymmetry`, `smoothness`
- `sersic_n`, `ellipticity`, `elongation`
- Various shape and texture measurements

Load results:
```python
import numpy as np

# Load morphology results
morph_z, morph_joint, morph_ha = np.load('M{objectid}_morph.npy', allow_pickle=True)

# Access measurements
print(f"H-alpha Gini: {morph_ha.gini}")
print(f"z-band M20: {morph_z.m20}")
print(f"Joint asymmetry: {morph_joint.asymmetry}")
```

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER | grep hamorph
```

### View logs
```bash
# Logs are in /scratch/gpfs/$USER/hamorph_logs/
ls /scratch/gpfs/$USER/hamorph_logs/

# View specific chunk output
tail -f /scratch/gpfs/$USER/hamorph_logs/hamorph_{JOBID}_{ARRAYID}.out

# Check for errors
grep -i error /scratch/gpfs/$USER/hamorph_logs/hamorph_{JOBID}_*.err
```

### Count completed galaxies
```bash
# Count output files
ls -1 /path/to/output/*.npy | wc -l
```

## Resource Requirements

Per chunk (default settings):
- **CPUs**: 4 cores
- **Memory**: 8 GB
- **Time**: 2 hours (conservative)
- **Partition**: CPU

These can be adjusted in `submit_hamorph.slurm` if needed.

## Troubleshooting

### Script fails to find cutouts

**Issue**: Many galaxies skipped with "No cutouts found" messages

**Solution**:
1. Verify `image_path` in config points to correct directory
2. Check that cutout directories exist: `{image_path}/hsc/` and `{image_path}/merian/`
3. Verify cutout file naming matches convention (uses `carpenter.conventions.produce_merianobjectname()`)

### Memory errors

**Issue**: Job killed with "Out of memory" errors

**Solution**: Increase memory in SLURM script:
```bash
#SBATCH --mem=16G  # or higher
```

### No sources found in segmentation map

**Issue**: Some galaxies fail with "No sources found" message

**Solution**: This is expected for low S/N objects. They are automatically skipped and logged.

### Different results across chunks

**Issue**: Inconsistent morphology measurements

**Solution**: This should not happen - each galaxy is processed independently. If you see this:
1. Check for version differences in dependencies
2. Verify all chunks use same config file
3. Check for filesystem issues (corruption, incomplete writes)

## Advanced Usage

### Custom chunk ranges

Process specific subsets:
```bash
# Only process chunks 10-20
sbatch --array=10-20 submit_hamorph.slurm

# Non-contiguous chunks
sbatch --array=0,5,10,15,20 submit_hamorph.slurm
```

### Different configurations per run

```bash
# Create custom config
cp ../configs/hamorph_config.yaml ../configs/hamorph_zscale.yaml
# Edit hamorph_zscale.yaml to set ha_im_method: zscale

# Submit with custom config
sbatch --array=0-49 submit_hamorph.slurm ../configs/hamorph_zscale.yaml
```

### Reprocessing failed chunks

After a job array completes, check which chunks failed and resubmit:

```bash
# Find chunks with errors
grep -l "FAILURE" /scratch/gpfs/$USER/hamorph_logs/hamorph_{JOBID}_*.out

# Resubmit specific failed chunks
sbatch --array=3,7,15 submit_hamorph.slurm
```

## Technical Details

### Chunking Strategy

When running as a SLURM array job:
1. Script loads full catalog with specified mask
2. Filters to only galaxies with available cutouts (17 in test case)
3. Divides filtered catalog into `n_chunks` equal parts
4. Each array task processes one chunk based on `SLURM_ARRAY_TASK_ID`

Example with 17 galaxies and 50 chunks:
- Most chunks process 0 galaxies (skipped)
- First 17 chunks each process 1 galaxy
- This is inefficient for small test sets but works well for large catalogs

**Recommendation**: For production, adjust `n_chunks` based on catalog size:
- Small datasets (< 100 galaxies): Use `n_chunks = 10` or fewer
- Medium datasets (100-1000): Use `n_chunks = 20-50`
- Large datasets (> 1000): Use `n_chunks = 50-100`

### Dependencies

Required Python packages:
- `numpy`, `pandas`, `astropy`, `scipy`
- `carpenter` (Merian photometry package)
- `pieridae` (Merian starburst analysis)
- `statmorph_joint` (morphology measurements)
- `sep` (source extraction)
- `pyyaml`

Cluster modules:
- `anaconda3/2023.9`
- Conda environment: `merenv` (or customize in scripts)

## Contact

For issues or questions:
- Check existing logs in `/scratch/gpfs/$USER/hamorph_logs/`
- Review config file settings
- Test locally with `--verbose` flag before cluster submission
