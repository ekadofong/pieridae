# Workflow: Flagging and Classifying Objects with Insufficient Labels

This workflow uses BYOL embeddings and k-nearest neighbors to identify galaxies that need manual classification, then provides a GUI for efficient visual inspection.

## Overview

1. **flag_for_classification.py**: Analyzes BYOL embeddings to find objects with fewer than 5 labeled neighbors in PCA space
2. **visual_inspection_gui.py**: GUI for manually classifying the flagged objects

## Prerequisites

- Trained BYOL model (`byol_final_model.pt` or `model_checkpoint.pt`)
- BYOL configuration file (`byol_config.yaml`)
- Image data in pickle format (`M*/[gi]_results.pkl`)
- Existing classification labels (CSV file specified in config)
- Python packages: `torch`, `numpy`, `pandas`, `scikit-learn`, `byol-pytorch`, `umap-learn`, `tkinter`, `PIL`

## Step 1: Flag Objects for Classification

Run the flagging script to identify objects that need manual classification:

```bash
cd /Users/kadofong/work/projects/merian/pieridae/scripts/byol
python flag_for_classification.py
```

### What it does:
1. Loads images from the configured data path
2. Extracts BYOL embeddings using the trained model
3. Applies PCA dimensionality reduction (default: 50 components)
4. Finds 50 nearest neighbors for each object in PCA space
5. Counts labeled neighbors for each object
6. Identifies objects with `n_labels < 5`
7. Saves flagged object IDs to CSV

### Output:
- **File**: `../../local_data/byol_results/flagged_for_classification.csv`
- **Format**: Single column CSV with header `object_id` containing object IDs like `M123456`

### Example output:
```
🔍 Loading image data from: ../../local_data/pieridae_output/starlet/msorabove_v0
📸 Found 22913 image files
✅ Loaded 22913 images
🧠 Extracting embeddings from 22913 images...
🔄 Computing PCA...
✅ PCA components: 50
✅ Explained variance: 97.6%
🔍 Finding 50 nearest neighbors in PCA space...
✅ 6199 objects have auto-labels
✅ 16714 objects have fewer than 5 labeled neighbors
💾 Saved 16714 flagged object IDs to: ../../local_data/byol_results/flagged_for_classification.csv
```

## Step 2: Visually Classify Flagged Objects

Use the visual inspection GUI to classify the flagged objects:

```bash
cd /Users/kadofong/work/surveys/merian/quick_projects/vizinspect/scripts
python visual_inspection_gui.py \
    --source msorabove_v0 \
    --flagged /Users/kadofong/work/projects/merian/pieridae/local_data/byol_results/flagged_for_classification.csv
```

### Arguments:
- `--source`: Source tag for images (must match directory in `fig_source/`)
- `--flagged`: Path to the CSV file from Step 1

### What it does:
1. Loads flagged object IDs from CSV
2. Filters image list to show only flagged objects
3. Displays images one at a time for classification
4. Saves classifications to both:
   - Session file: `classifications_{source}_{user}_{date}.csv`
   - Base file: `classifications_kadofong_current.csv`

### Classification categories:
- **1**: Undisturbed (green)
- **2**: Ambiguous (orange)
- **3**: Merger (red)
- **4**: Fragmentation (blue)
- **5**: Artifact (purple)

### Keyboard shortcuts:
- `1-5`: Classify current object
- `←/→`: Navigate between objects
- Auto-advances to next object after classification

### Output:
- **Session file**: `classifications_{source}_{user}_{date}.csv`
- **Base file**: `classifications_kadofong_current.csv` (merged with existing classifications)

## Step 3: Re-run Analysis (Optional)

After classifying more objects, you can re-run the flagging script to identify remaining objects that still need classification:

```bash
cd /Users/kadofong/work/projects/merian/pieridae/scripts/byol
python flag_for_classification.py
```

The script will use the updated classifications and potentially flag fewer objects as more labels become available.

## Configuration

### BYOL Config (`byol_config.yaml`)

Key parameters that affect flagging:

```yaml
data:
  input_path: "../../local_data/pieridae_output/starlet/msorabove_v0"
  output_path: "../../local_data/byol_results"

labels:
  classifications_file: "./classifications_kadofong_20250929.csv"

analysis:
  pca_components: null  # Auto-determine based on variance threshold
  explained_variance_threshold: 0.95
```

### Neighbor Threshold

To change the threshold for flagging (currently `n_labels < 5`), edit line 296 in `flag_for_classification.py`:

```python
# Change threshold here (e.g., to flag objects with < 3 labeled neighbors)
prob_labels[n_labels < 5] = 0.
```

## File Locations

### Input Files:
- BYOL model: `../../local_data/byol_results/byol_final_model.pt`
- Config: `./byol_config.yaml`
- Images: `../../local_data/pieridae_output/starlet/msorabove_v0/M*/[gi]_results.pkl`
- Labels: Path specified in config (e.g., `./classifications_kadofong_20250929.csv`)

### Output Files:
- Flagged objects: `../../local_data/byol_results/flagged_for_classification.csv`
- Classifications (session): `/Users/kadofong/work/surveys/merian/quick_projects/vizinspect/scripts/classifications_{source}_{user}_{date}.csv`
- Classifications (base): `/Users/kadofong/work/surveys/merian/quick_projects/vizinspect/scripts/classifications_kadofong_current.csv`

## Workflow Tips

1. **Iterative classification**: Run the flagging script → classify objects → re-run flagging to see progress
2. **Multiple sessions**: The GUI automatically merges classifications from different sessions into `classifications_kadofong_current.csv`
3. **Skip already classified**: The GUI starts at the first unclassified object in the filtered list
4. **Check progress**: The GUI shows progress (e.g., "150 / 500") in the top right

## Troubleshooting

### "No files found matching pattern"
- Check that `input_path` in config points to correct directory
- Verify pickle files exist with pattern `M*/[gi]_results.pkl`

### "No trained model found"
- Ensure `byol_final_model.pt` exists in output path
- Or that `model_checkpoint.pt` exists as fallback

### "Label file not found"
- Update `classifications_file` path in `byol_config.yaml`
- Use absolute or relative path from script directory

### GUI shows no images
- Verify `--source` matches a directory in `fig_source/`
- Check that PNG files exist for the flagged object IDs
- Confirm flagged CSV has correct format with `object_id` column

### MPS/GPU errors
- Script automatically falls back to CPU for problematic batches
- Reduce batch size in config if needed

## Example Complete Workflow

```bash
# 1. Flag objects needing classification
cd /Users/kadofong/work/projects/merian/pieridae/scripts/byol
python flag_for_classification.py

# Output: 16714 objects flagged

# 2. Classify flagged objects
cd /Users/kadofong/work/surveys/merian/quick_projects/vizinspect/scripts
python visual_inspection_gui.py \
    --source starlet_starbursts \
    --flagged /Users/kadofong/work/projects/merian/local_data/byol_results/flagged_for_classification.csv

# (Classify objects in GUI, then close when done)

# 3. Check progress - re-run flagging
cd /Users/kadofong/work/projects/merian/pieridae/scripts/byol
python flag_for_classification.py

# Output: Fewer objects flagged as more labels are available

# 4. Repeat steps 2-3 until satisfied with coverage
```
