#!/usr/bin/env python3
"""
Flag Objects for Manual Classification

Refactored version using pieridae.starbursts.byol module.
Identifies objects with insufficient labeled neighbors and flags them for manual classification.

Usage
-----
# Use default config
python flag_objects.py

# Custom config
python flag_objects.py --config custom_config.yaml

# Custom threshold
python flag_objects.py --n-min 8
"""

import sys
import argparse
import logging
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import pickle

# Add pieridae to path (go up 2 levels: scripts/ -> merger_analysis/ -> pieridae/)
sys.path.insert(0, str(Path(__file__).parents[2]))

from pieridae.starbursts.byol import (
    BYOLModelManager,
    EmbeddingAnalyzer,
    LabelPropagation,
    load_merian_images
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['data']['input_path'] = Path(config['data']['input_path'])
    config['data']['output_path'] = Path(config['data']['output_path'])

    return config


def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('flag_objects')
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_labels(config: dict, img_names: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Load classification labels"""
    label_file = Path(config.get('labels', {}).get('classifications_file', ''))

    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    logger.info(f"Loading labels from: {label_file}")

    mergers = pd.read_csv(label_file, index_col=0)
    labels = mergers.reindex(img_names)
    labels = labels.replace(np.nan, 0).values.flatten().astype(int)

    logger.info(f"Loaded {len(labels)} labels")

    # Print label distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_meanings = config.get('labels', {}).get('label_mapping', {})

    logger.info("Label distribution:")
    for label_val, count in zip(unique, counts):
        meaning = label_meanings.get(label_val, f"unknown_{label_val}")
        logger.info(f"   {label_val} ({meaning}): {count} objects")

    return labels


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Flag objects for manual classification based on label propagation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script:
1. Loads trained BYOL model and extracts embeddings
2. Computes PCA transformation
3. Performs iterative K-NN label propagation
4. Flags objects with fewer than n_min labeled neighbors for manual classification

Output:
- flagged_for_classification.csv: List of object IDs needing classification
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='../config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--n-min',
        type=int,
        help='Minimum labeled neighbors threshold (overrides config)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output CSV filename (default: flagged_for_classification.csv)'
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("OBJECT FLAGGING FOR CLASSIFICATION")
    logger.info("=" * 60)

    output_path = config['data']['output_path']
    output_path.mkdir(parents=True, exist_ok=True)

    # Get parameters
    n_min = args.n_min or config.get('labels', {}).get('minimum_labeled_neighbors', 8)
    n_min_auto = config.get('labels', {}).get('minimum_labeled_neighbors_for_autoprop', 15)
    n_neighbors = config.get('labels', {}).get('n_neighbors', 50)
    prob_threshold = config.get('labels', {}).get('prob_threshold', 0.6)
    frag_threshold = config.get('labels', {}).get('frag_threshold', 0.1)

    logger.info(f"Using minimum_labeled_neighbors = {n_min}")
    logger.info(f"Using minimum_labeled_neighbors_for_autoprop = {n_min_auto}")
    logger.info(f"Using n_neighbors = {n_neighbors}")

    try:
        # Load images
        logger.info("Loading images...")
        images, img_names = load_merian_images(
            config['data']['input_path'],
            logger
        )

        # Load/extract embeddings
        embeddings_file = output_path / 'embeddings.npy'
        if embeddings_file.exists():
            logger.info("Loading existing embeddings...")
            embeddings = np.load(embeddings_file)
        else:
            logger.info("Extracting embeddings...")
            model_manager = BYOLModelManager(config, output_path, logger)
            embeddings = model_manager.extract_embeddings(images)

        # Compute PCA
        logger.info("Computing PCA...")
        analyzer = EmbeddingAnalyzer(config, logger)
        embeddings_pca = analyzer.compute_pca(embeddings)

        # Load labels
        labels = load_labels(config, img_names, logger)

        # Label propagation
        logger.info("Running iterative label propagation...")
        propagator = LabelPropagation(
            n_neighbors=n_neighbors,
            n_min=n_min,
            n_min_auto=n_min_auto,
            prob_threshold=prob_threshold,
            frag_threshold=frag_threshold,
            logger=logger
        )

        iterative_labels, n_labels_iter, prob_labels_iter, stats = \
            propagator.iterative_propagation(embeddings_pca, labels)

        # Flag objects
        output_file = args.output_file or 'flagged_for_classification.csv'
        output_file = output_path / output_file

        flagged_objects = propagator.flag_objects_for_classification(
            img_names,
            n_labels_iter,
            output_file,
            n_min=n_min
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total objects: {len(img_names)}")
        logger.info(f"Human labels: {stats['n_human']}")
        logger.info(f"Auto-labels added (iteration 1): {stats['n_added_iteration']}")
        logger.info(f"Total labels after iteration: {stats['n_total_after_iteration']}")
        logger.info(f"Objects with auto-labels (iteration 2): {stats['n_final_auto']}")
        logger.info(f"Flagged for classification: {len(flagged_objects)}")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 60)

        print(f"\n✅ Successfully flagged {len(flagged_objects)} objects")
        print(f"   Output saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
