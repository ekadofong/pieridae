#!/usr/bin/env python3
"""
Minimal Working Example: Load FIRE-2 Particle Data

This script demonstrates the minimal setup needed to load particle data
from a FIRE-2 simulation snapshot using gizmo_analysis.
"""

import sys
from pathlib import Path

# Add pieridae to path (adjust if needed)
sys.path.insert(0, str(Path(__file__).parents[2]))

# Import gizmo_analysis
try:
    import gizmo_analysis as gizmo
except ImportError:
    raise ImportError(
        "gizmo_analysis not available. Install with:\n"
        "  pip install git+https://github.com/lzkelley/gizmo_analysis.git"
    )


def load_fire2_snapshot(
        tag = "m11d_res7100",  # Galaxy identifier
        snapshot_id = 600,  # Snapshot number
        particle_types = ["star"],  # Particle types to load: 'star', 'gas', 'dark', etc.
    ):
    """
    Load particle data from a FIRE-2 simulation snapshot.

    Returns
    -------
    particle_data : dict
        Particle data dictionary from gizmo_analysis
    """
    # Configuration
    fire2_base_path = Path(__file__).parents[2] / "local_data" / "fire2" / "core"


    # Build full simulation directory path
    simulation_directory = str(fire2_base_path / tag)

    print(f"Loading FIRE-2 snapshot...")
    print(f"  Directory: {simulation_directory}")
    print(f"  Snapshot: {snapshot_id}")
    print(f"  Particle types: {particle_types}")

    # Load particle data
    # This is the key function call from gizmo_analysis
    particle_data = gizmo.io.Read.read_snapshots(
        particle_types,      # List of particle types to load
        'index',             # Snapshot index format ('index', 'time', or 'redshift')
        snapshot_id,         # Snapshot ID to load
        simulation_directory,  # Path to simulation directory
        assign_hosts=True,   # Compute host galaxy properties
        assign_hosts_rotation=False  # Don't compute rotation curves
    )

    # Print summary information
    print(f"\nLoaded successfully!")
    print(f"  N_particles: {len(particle_data[particle_types[0]]['position'])}")
    print(f"  Host position: {particle_data.host['position'][0]} [comoving kpc]")
    print(f"  Host velocity: {particle_data.host['velocity'][0]} [km/s]")
    print(f"  Redshift: {particle_data.snapshot['redshift']}")
    print(f"  Scale factor: {particle_data.snapshot['scalefactor']}")

    # Available particle data keys
    print(f"\nAvailable particle data keys:")
    for key in sorted(particle_data[particle_types[0]].keys()):
        data_shape = particle_data[particle_types[0]][key].shape
        print(f"  {key}: {data_shape}")

    return particle_data


if __name__ == "__main__":
    # Load the data
    particle_data = load_fire2_snapshot()

    # Example: Access particle positions and masses
    positions = particle_data["star"]["position"]  # Shape: (N_particles, 3)
    masses = particle_data["star"]["mass"]  # Shape: (N_particles,)

    print(f"\nExample usage:")
    print(f"  First particle position: {positions[0]} [comoving kpc]")
    print(f"  First particle mass: {masses[0]:.2e} [Msun]")
