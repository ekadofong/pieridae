# FIRE2 Mock Image Generation

Script for generating randomly oriented projection images from FIRE2 simulation snapshots.

## Overview

`generate_fire2_images.py` creates N projection images of FIRE2 galaxies with random viewing angles. Each image is a 2D histogram of particle positions with configurable bin sizes and physical extents.

## Features

- **Random 3D rotations**: Uniform sampling of viewing angles using random rotation matrices
- **Fibonacci sphere option**: More evenly distributed viewing angles for systematic coverage
- **Flexible binning**: Configurable physical size and bin width (default: 1 kpc bins, 150 kpc extent)
- **Multiple particle types**: Support for star, gas, and dark matter particles
- **Optional noise**: Add Gaussian or Poisson noise to simulate observations
- **Metadata tracking**: Save rotation matrices and parameters for reproducibility
- **Multiple output formats**: Save as NumPy arrays (.npy) or PNG images

## Installation Requirements

```bash
# Required packages
pip install numpy scipy pyyaml tqdm

# For FIRE2 data reading (required)
# Download from: https://bitbucket.org/awetzel/gizmo_analysis
# Add to PYTHONPATH

# Optional (for PNG output)
pip install matplotlib
```

## Quick Start

### 1. Basic Usage (Default Config)

```bash
cd merger_analysis/scripts
python generate_fire2_images.py
```

This uses the default configuration at `../configs/fire2_image_config.yaml`

### 2. Test with Small Example

```bash
# Generate just 3 images for testing
python generate_fire2_images.py --config ../configs/fire2_image_test.yaml
```

### 3. Generate Images for Different Galaxies

```bash
# m11b_res2100
python generate_fire2_images.py --tag m11b_res2100 --n-images 100

# m11d_res7100
python generate_fire2_images.py --tag m11d_res7100 --n-images 100

# Different snapshot
python generate_fire2_images.py --tag m11b_res2100 --snapshot 500
```

### 4. Add Noise

```bash
# Add Gaussian noise
python generate_fire2_images.py --add-noise --n-images 50
```

## Configuration

Configuration files are in YAML format. See `configs/fire2_image_config.yaml` for the full template.

### Key Parameters

#### Simulation
```yaml
simulation:
  tag: m11b_res2100              # Galaxy identifier
  snapshot_id: 600               # Snapshot number
  fire2_base_path: ../local_data/fire2/core/
  particle_types: [star]         # Particle types to project
```

#### Projection
```yaml
projection:
  n_images: 100                  # Number of random projections
  image_size_kpc: 150            # Physical extent in kpc
  bin_width_kpc: 1.0             # Histogram bin size in kpc
  max_distance_kpc: 100          # Particle selection radius
  log_scale: true                # Apply log10 to histogram values
```

#### Noise
```yaml
noise:
  add_noise: false               # Enable noise
  noise_type: gaussian           # gaussian or poisson
  gaussian_sigma: 0.1            # Noise level (fraction of mean)
```

#### Output
```yaml
output:
  output_dir: ../../local_data/mock_images/
  save_format: npy               # npy, png, or both
  save_metadata: true            # Save rotation matrices
  random_seed: 42                # For reproducibility
```

## Output Structure

```
local_data/mock_images/
└── {tag}/                       # e.g., m11b_res2100/
    ├── image_0000.npy          # First projection
    ├── image_0001.npy          # Second projection
    ├── ...
    ├── image_0099.npy          # Last projection
    └── metadata.json            # Metadata with rotation matrices
```

### Metadata Format

```json
{
  "config": {...},                    # Full configuration used
  "simulation_directory": "...",      # Path to simulation
  "snapshot_id": 600,                 # Snapshot number
  "host_center": [x, y, z],          # Galaxy center (comoving kpc)
  "n_particles": 952466,              # Number of particles
  "n_images": 100,                    # Number of images generated
  "images": [
    {
      "index": 0,                     # Image index
      "filename": "image_0000",       # Output filename
      "rotation_matrix": [[...], [...], [...]],  # 3x3 rotation
      "image_shape": [150, 150],      # Array dimensions
      "rotation_method": "uniform"    # Method used
    },
    ...
  ]
}
```

## Command-Line Options

```
--config CONFIG          Path to config YAML file
--n-images N            Override number of images
--tag TAG               Override galaxy tag
--snapshot SNAPSHOT     Override snapshot ID
--output-dir DIR        Override output directory
--add-noise             Enable noise (overrides config)
--no-verbose            Reduce verbosity
```

## Algorithm Details

### Random Rotation Generation

For each image:

1. **Generate random Euler angles** (α, β, γ):
   - α, γ: uniform in [0, 2π]
   - β: arccos(uniform(-1, 1)) for uniform sphere sampling

2. **Create rotation matrix** from Euler angles (ZYZ convention)

3. **Apply rotation** to particle positions:
   ```
   pos_rotated = R @ (pos - host_center)
   ```

4. **Project** onto x-y plane (use first two coordinates after rotation)

5. **Create 2D histogram** with specified binning

6. **Apply log scaling** (optional):
   ```
   image = log10(histogram + log_min_value)
   ```

### Fibonacci Sphere Method

Set `rotation_method: fibonacci` for more uniform angular coverage. This uses the Fibonacci lattice on a sphere to generate evenly spaced viewing directions.

## Examples

### Example 1: Generate Training Set

```bash
# Generate 500 images for BYOL training
python generate_fire2_images.py \
    --tag m11b_res2100 \
    --n-images 500 \
    --output-dir /path/to/training_data
```

### Example 2: Multiple Galaxies

```bash
# Create a loop for multiple galaxies
for tag in m11b_res2100 m11d_res7100 m11h_res7100; do
    python generate_fire2_images.py \
        --tag $tag \
        --n-images 100
done
```

### Example 3: Include Gas Particles

Edit config to add gas:
```yaml
simulation:
  particle_types:
    - star
    - gas
```

### Example 4: Custom Physical Size

```bash
# Larger images (300 kpc, 2 kpc bins)
python generate_fire2_images.py \
    --config custom_large.yaml
```

With `custom_large.yaml`:
```yaml
projection:
  image_size_kpc: 300
  bin_width_kpc: 2.0
  max_distance_kpc: 200
```

## Loading Generated Images

### In Python

```python
import numpy as np
import json

# Load metadata
with open('local_data/mock_images/m11b_res2100/metadata.json', 'r') as f:
    metadata = json.load(f)

# Load specific image
image = np.load('local_data/mock_images/m11b_res2100/image_0000.npy')

# Get rotation matrix for that image
rotation_matrix = np.array(metadata['images'][0]['rotation_matrix'])

print(f"Image shape: {image.shape}")
print(f"Image range: [{image.min():.2f}, {image.max():.2f}]")
```

### Visualize

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(image.T, origin='lower', cmap='viridis')
ax.set_xlabel('X (kpc)')
ax.set_ylabel('Y (kpc)')
plt.colorbar(im, label='log10(particle count)')
plt.show()
```

## Troubleshooting

### Error: "gizmo_analysis not available"
- Install gizmo_analysis package
- Add to PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:/path/to/gizmo_analysis`

### Error: "No particles within max_distance_kpc"
- Increase `max_distance_kpc` in config
- Check that `fire2_base_path` points to correct directory
- Verify snapshot file exists

### Images look empty
- Check `log_scale` and `log_min_value` settings
- Try setting `log_scale: false` to see raw histogram
- Verify particle data loaded correctly (check console output)

### Memory issues
- Reduce `n_images` or generate in batches
- Use smaller `max_distance_kpc` to load fewer particles
- Set `cache_particle_data: false` in advanced config

## Integration with BYOL Pipeline

Generated images can be used directly with the BYOL training pipeline:

```python
from pieridae.starbursts.byol import BYOLModelManager

# Load generated images
images = []
for i in range(n_images):
    img = np.load(f'local_data/mock_images/{tag}/image_{i:04d}.npy')
    images.append(img)
images = np.array(images)

# Add channel dimension if needed
if images.ndim == 3:
    images = images[:, np.newaxis, :, :]  # (N, 1, H, W)

# Train BYOL
model_manager = BYOLModelManager(config, output_path, logger)
model_manager.train_model(images)
```

## Citation

If using this code, please cite:
- FIRE-2 simulations: Hopkins et al. (2018)
- gizmo_analysis: Wetzel et al.

## Author

Created for the Merian Survey pieridae pipeline.
Based on exploratory analysis in `explore_simulations.ipynb`.
