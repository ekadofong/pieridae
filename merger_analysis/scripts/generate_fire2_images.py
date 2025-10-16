#!/usr/bin/env python3
"""
FIRE2 Mock Image Generation Script

Generates N randomly oriented projection images from FIRE2 simulation snapshots.
Based on the exploratory analysis in explore_simulations.ipynb.

Usage
-----
# Use default config
python generate_fire2_images.py

# Use custom config
python generate_fire2_images.py --config ../configs/custom_config.yaml

# Override parameters
python generate_fire2_images.py --n-images 500 --tag m11d_res7100

# Different snapshot
python generate_fire2_images.py --tag m11b_res2100 --snapshot 600
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import yaml
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from ekfplot import plot as ek

# Add pieridae to path
sys.path.insert(0, str(Path(__file__).parents[2]))

try:
    import gizmo_analysis as gizmo
    GIZMO_AVAILABLE = True
except ImportError:
    GIZMO_AVAILABLE = False
    print("WARNING: gizmo_analysis not available. Please install it to use this script.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('fire2_images')
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_random_rotation_matrix(random_state: np.random.RandomState) -> np.ndarray:
    """
    Generate a random 3D rotation matrix using uniformly distributed Euler angles.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random state for reproducibility

    Returns
    -------
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    """
    # Generate random Euler angles
    # Use uniform distribution for proper coverage of rotation space
    alpha = random_state.uniform(0, 2 * np.pi)  # Rotation about z-axis
    beta = np.arccos(random_state.uniform(-1, 1))  # Rotation about y-axis (uniform on sphere)
    gamma = random_state.uniform(0, 2 * np.pi)  # Rotation about z-axis again

    # Create rotation from Euler angles (ZYZ convention)
    rotation = Rotation.from_euler('zyz', [alpha, beta, gamma])
    return rotation.as_matrix()


def generate_fibonacci_rotation_matrices(n_rotations: int) -> List[np.ndarray]:
    """
    Generate evenly distributed rotation matrices using Fibonacci sphere.

    This provides more uniform coverage of rotation space than random sampling.

    Parameters
    ----------
    n_rotations : int
        Number of rotation matrices to generate

    Returns
    -------
    rotation_matrices : List[np.ndarray]
        List of 3x3 rotation matrices
    """
    rotation_matrices = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

    for i in range(n_rotations):
        y = 1 - (i / float(n_rotations - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        # Create viewing direction
        viewing_direction = np.array([x, y, z])

        # Create rotation matrix to align z-axis with viewing direction
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, viewing_direction)

        if np.linalg.norm(rotation_axis) > 1e-10:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, viewing_direction), -1, 1))
            rotation = Rotation.from_rotvec(angle * rotation_axis)
            rotation_matrices.append(rotation.as_matrix())
        else:
            # Viewing direction is already along z-axis
            rotation_matrices.append(np.eye(3))

    return rotation_matrices


def load_particle_data(
    simulation_directory: str,
    snapshot_id: int,
    particle_types: List[str],
    logger: logging.Logger
) -> dict:
    """
    Load particle data from FIRE2 simulation.

    Parameters
    ----------
    simulation_directory : str
        Path to simulation directory
    snapshot_id : int
        Snapshot number to load
    particle_types : List[str]
        List of particle types to load (e.g., ['star', 'gas'])
    logger : logging.Logger
        Logger instance

    Returns
    -------
    particle_data : dict
        Particle data dictionary from gizmo_analysis
    """
    if not GIZMO_AVAILABLE:
        raise ImportError("gizmo_analysis package is required but not available")

    logger.info(f"Loading particle data from: {simulation_directory}")
    logger.info(f"  Snapshot ID: {snapshot_id}")
    logger.info(f"  Particle types: {particle_types}")

    # Read snapshot
    particle_data = gizmo.io.Read.read_snapshots(
        particle_types,
        'index',
        snapshot_id,
        simulation_directory,
        assign_hosts=True,
        assign_hosts_rotation=False
    )

    logger.info(f"Loaded {len(particle_data[particle_types[0]]['position'])} particles")
    logger.info(f"Host position: {particle_data.host['position'][0]}")

    return particle_data


def create_projection_image(
    positions: np.ndarray,
    rotation_matrix: np.ndarray,
    host_center: np.ndarray,
    image_size_pix: float,
    bin_width_kpc: float,
    max_distance_kpc: float,
    log_scale: bool = True,
    log_min_value: float = 1.0,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D projection image of particles.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions (N x 3) in comoving kpc
    rotation_matrix : np.ndarray
        3x3 rotation matrix
    host_center : np.ndarray
        Center position of host galaxy (comoving kpc)
    image_size_pix : int
        Physical size of image in pixels
    bin_width_kpc : float
        Size of histogram bins in kpc
    max_distance_kpc : float
        Maximum distance from center to include particles
    log_scale : bool
        Apply log10 scaling to histogram
    log_min_value : float
        Minimum value for log scaling
    weights : Optional[np.ndarray]
        Weights for particles (e.g., masses)

    Returns
    -------
    image : np.ndarray
        2D histogram image
    xedges : np.ndarray
        Bin edges for x-axis
    yedges : np.ndarray
        Bin edges for y-axis
    """
    # Center on host
    centered_positions = positions - host_center

    # Filter by distance
    distances = np.linalg.norm(centered_positions, axis=1)
    in_view = distances < max_distance_kpc

    if in_view.sum() == 0:
        raise ValueError("No particles within max_distance_kpc of host center")

    centered_positions = centered_positions[in_view]
    if weights is not None:
        weights = weights[in_view]

    # Apply rotation
    rotated_positions = centered_positions @ rotation_matrix.T

    # Project onto x-y plane (after rotation)
    x_proj = rotated_positions[:, 0]
    y_proj = rotated_positions[:, 1]

    # Create histogram bins
    #n_bins = int(image_size_kpc / bin_width_kpc)
    #bins = np.linspace(-image_size_kpc / 2, image_size_kpc / 2, n_bins + 1)
    
    image_size_kpc = image_size_pix * bin_width_kpc
    n_bins = image_size_pix 
    bins = np.linspace(-image_size_kpc / 2, image_size_kpc / 2, n_bins + 1)

    # Create 2D histogram
    image, xedges, yedges = np.histogram2d(
        x_proj,
        y_proj,
        bins=[bins, bins],
        weights=weights
    )

    # Apply log scaling if requested
    if log_scale:
        image = np.log10(image + log_min_value)

    return image, xedges, yedges


def add_noise_to_image(
    image: np.ndarray,
    noise_type: str,
    gaussian_sigma: float = 0.1,
    gaussian_absolute: bool = False,
    poisson_scale: float = 1.0,
    random_state: np.random.RandomState = None
) -> np.ndarray:
    """
    Add noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image
    noise_type : str
        Type of noise: 'gaussian' or 'poisson'
    gaussian_sigma : float
        Standard deviation for Gaussian noise
    gaussian_absolute : bool
        If True, sigma is absolute; if False, relative to image mean
    poisson_scale : float
        Scaling factor for Poisson noise
    random_state : np.random.RandomState
        Random state for reproducibility

    Returns
    -------
    noisy_image : np.ndarray
        Image with added noise
    """
    if random_state is None:
        random_state = np.random.RandomState()

    if noise_type == 'gaussian':
        if gaussian_absolute:
            sigma = gaussian_sigma
        else:
            sigma = gaussian_sigma * np.mean(image[image>0])
        noise = random_state.normal(0, sigma, image.shape)
        
        return image + noise

    elif noise_type == 'poisson':
        # Scale image to reasonable count levels
        scaled_image = image * poisson_scale
        # Apply Poisson noise
        noisy_scaled = random_state.poisson(scaled_image)
        # Scale back
        return noisy_scaled / poisson_scale

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def save_image(
    image: np.ndarray,
    output_path: Path,
    save_format: str = 'npy',
    normalize: bool = False
) -> None:
    """
    Save image to disk.

    Parameters
    ----------
    image : np.ndarray
        Image to save
    output_path : Path
        Output file path (without extension)
    save_format : str
        Format: 'npy', 'png', or 'both'
    normalize : bool
        Normalize image to [0, 1] range for PNG
    """
    if normalize:
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
    else:
        image_norm = image

    if save_format in ['npy', 'both']:
        np.save(output_path.with_suffix('.npy'), image)

    if save_format in ['png', 'both'] and PLOTTING_AVAILABLE:
        ek.imshow(
            image_norm,
            cmap='Greys'
        )
        plt.savefig(output_path.with_suffix('.png'))
        plt.close ()



def generate_images(
    config: dict,
    logger: logging.Logger
) -> Dict:
    """
    Main function to generate mock images.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance

    Returns
    -------
    metadata : dict
        Metadata about the generated images
    """
    # Extract config parameters
    sim_config = config['simulation']
    proj_config = config['projection']
    noise_config = config['noise']
    output_config = config['output']
    advanced_config = config.get('advanced', {})

    # Setup random state
    random_seed = output_config.get('random_seed', 42)
    random_state = np.random.RandomState(random_seed)

    # Build simulation directory path
    simulation_directory = str(Path(sim_config['fire2_base_path']) / sim_config['tag'])

    # Load particle data
    particle_data = load_particle_data(
        simulation_directory,
        sim_config['snapshot_id'],
        sim_config['particle_types'],
        logger
    )

    # Get particle positions and optional weights
    particle_type = sim_config['particle_types'][0]  # Use first type for now
    positions = particle_data[particle_type]['position']
    host_center = particle_data.host['position'][0]

    # Use mass as weights if available
    weights = particle_data[particle_type].get('mass', None)

    # Setup output directory
    output_base = Path(output_config['output_dir'])
    output_dir = output_base / sim_config['tag']
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Handle append mode
    append_mode = output_config.get('append_mode', False)
    starting_index = 0
    existing_metadata = None

    if append_mode:
        metadata_path = output_dir / 'metadata.json'
        if metadata_path.exists():
            logger.info("Append mode: Loading existing metadata...")
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            # Find the highest existing image index for this snapshot
            snapshot_id = sim_config['snapshot_id']
            tag = sim_config['tag']
            max_index = -1
            for img_meta in existing_metadata.get('images', []):
                fname = img_meta.get('filename', '')
                # Parse filename: {tag}_{snapshot_id}_{index:04d}
                if fname.startswith(f"{tag}_{snapshot_id}_"):
                    try:
                        idx_str = fname.split('_')[-1]
                        idx = int(idx_str)
                        max_index = max(max_index, idx)
                    except (ValueError, IndexError):
                        pass
            starting_index = max_index + 1
            logger.info(f"Starting from index {starting_index} (found {max_index + 1} existing images)")
        else:
            logger.info("Append mode: No existing metadata found, starting fresh")
    else:
        logger.info("Overwrite mode: Will replace any existing images")

    # Generate rotation matrices
    n_images = proj_config['n_images']
    rotation_method = advanced_config.get('rotation_method', 'uniform')

    if rotation_method == 'fibonacci':
        logger.info("Generating rotation matrices using Fibonacci sphere...")
        rotation_matrices = generate_fibonacci_rotation_matrices(n_images)
    else:
        logger.info("Generating random rotation matrices...")
        rotation_matrices = [
            generate_random_rotation_matrix(random_state)
            for _ in range(n_images)
        ]

    # Generate images
    logger.info(f"Generating {n_images} projection images...")

    # Initialize metadata
    if append_mode and existing_metadata:
        # Preserve existing metadata and add new images
        metadata = existing_metadata
        # Update basic metadata
        metadata['n_images'] = len(metadata.get('images', [])) + n_images
    else:
        metadata = {
            'config': config,
            'simulation_directory': simulation_directory,
            'snapshot_id': sim_config['snapshot_id'],
            'host_center': host_center.tolist(),
            'n_particles': len(positions),
            'n_images': n_images,
            'images': []
        }

    tag = sim_config['tag']
    snapshot_id = sim_config['snapshot_id']

    for i in tqdm(range(n_images), desc="Generating images"):
        image_index = starting_index + i
        rotation_matrix = rotation_matrices[i]

        # Create projection
        try:
            image, xedges, yedges = create_projection_image(
                positions,
                rotation_matrix,
                host_center,
                proj_config['image_size_pix'],
                proj_config['bin_width_kpc'],
                proj_config['max_distance_kpc'],
                proj_config.get('log_scale', False),
                proj_config.get('log_min_value', 1.0),
                weights=weights
            )

            # Add noise if requested
            if noise_config.get('add_noise', False):
                noise_seed = random_seed + noise_config.get('noise_seed_offset', 1000) + image_index
                noise_random_state = np.random.RandomState(noise_seed)

                image = add_noise_to_image(
                    image,
                    noise_config['noise_type'],
                    noise_config.get('gaussian_sigma', 0.1),
                    noise_config.get('gaussian_absolute', False),
                    noise_config.get('poisson_scale', 1.0),
                    noise_random_state
                )

            # Save image with new naming scheme: {tag}_{snapshot_id}_{image_index:04d}
            filename = f"{tag}_{snapshot_id}_{image_index:04d}"
            output_path = output_dir / filename
            save_image(
                image,
                output_path,
                output_config.get('save_format', 'npy'),
                output_config.get('normalize_images', False)
            )

            # Store metadata for this image
            image_metadata = {
                'index': image_index,
                'filename': filename,
                'snapshot_id': snapshot_id,
                'rotation_matrix': rotation_matrix.tolist(),
                'image_shape': image.shape,
                'rotation_method': rotation_method
            }

            metadata['images'].append(image_metadata)

        except Exception as e:
            logger.error(f"Error generating image {image_index}: {e}")
            continue

    # Save metadata
    if output_config.get('save_metadata', True):
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to: {metadata_path}")

    logger.info(f"Successfully generated {len(metadata['images']) - starting_index} new images")
    logger.info(f"Total images for {tag}: {len(metadata['images'])}")

    return metadata


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate randomly oriented projection images from FIRE2 simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python generate_fire2_images.py

  # Custom config
  python generate_fire2_images.py --config custom_config.yaml

  # Override parameters
  python generate_fire2_images.py --n-images 500 --tag m11d_res7100

  # Different snapshot
  python generate_fire2_images.py --tag m11b_res2100 --snapshot 600
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='../configs/fire2_image_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--n-images',
        type=int,
        help='Override number of images to generate'
    )
    parser.add_argument(
        '--tag',
        type=str,
        help='Override galaxy tag (e.g., m11b_res2100)'
    )
    parser.add_argument(
        '--snapshot',
        type=int,
        help='Override snapshot ID'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    parser.add_argument(
        '--add-noise',
        action='store_true',
        help='Add noise to images (overrides config)'
    )
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Reduce verbosity'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.n_images is not None:
        config['projection']['n_images'] = args.n_images
    if args.tag is not None:
        config['simulation']['tag'] = args.tag
    if args.snapshot is not None:
        config['simulation']['snapshot_id'] = args.snapshot
    if args.output_dir is not None:
        config['output']['output_dir'] = args.output_dir
    if args.add_noise:
        config['noise']['add_noise'] = True

    # Setup logging
    verbose = config.get('advanced', {}).get('verbose', True) and not args.no_verbose
    logger = setup_logging(verbose)

    logger.info("=" * 70)
    logger.info("FIRE2 MOCK IMAGE GENERATION")
    logger.info("=" * 70)
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Galaxy tag: {config['simulation']['tag']}")
    logger.info(f"Snapshot ID: {config['simulation']['snapshot_id']}")
    logger.info(f"Number of images: {config['projection']['n_images']}")
    if config['noise']['add_noise']:
        logger.info(f"Adding noise to simulated images")
    logger.info('='*70)
        

    try:
        # Generate images
        metadata = generate_images(config, logger)

        # Print summary
        logger.info("=" * 70)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Generated {len(metadata['images'])} images")
        logger.info(f"Output directory: {Path(config['output']['output_dir']) / config['simulation']['tag']}")
        logger.info("=" * 70)

        print(f"\n✅ Successfully generated {len(metadata['images'])} mock images!")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
