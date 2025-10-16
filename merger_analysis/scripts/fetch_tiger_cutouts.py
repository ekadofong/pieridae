#!/usr/bin/env python
"""
Fetch Cutouts from Tiger3-Sumire Cluster

This script fetches cutouts (both Merian and HSC) from the tiger3-sumire.princeton.edu
cluster for a given Merian ID or object name. It tries starbursts_v0 first, then
falls back to msorabove_v0 if files are not found.

Usage:
    python fetch_tiger_cutouts.py M123456  # Using Merian ID
    python fetch_tiger_cutouts.py J095618.67+030835.28  # Using object name
    python fetch_tiger_cutouts.py M123456 -o ./my_data/  # Specify output directory
"""

import os
import sys
import argparse
import subprocess
import re
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from carpenter import conventions
from pieridae.starbursts import sample


def load_config(config_path: str = None):
    """Load configuration from YAML file"""
    if config_path is None:
        # Default to figures_config.yaml in configs directory
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / 'configs' / 'figures_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def is_merian_id(identifier):
    """Check if the identifier is a Merian ID (M followed by digits)."""
    return re.match(r'^M\d+$', identifier) is not None


def is_object_name(identifier):
    """Check if the identifier is an object name (starts with J)."""
    return identifier.startswith('J')


def get_object_name(identifier, catalog):
    """
    Convert identifier to object name.

    Parameters
    ----------
    identifier : str
        Either a Merian ID (M[0-9]+) or object name (J*)
    catalog : pd.DataFrame
        Merian catalog

    Returns
    -------
    str
        Object name (J*)
    str or None
        Catalog ID (Merian ID) if available
    """
    if is_object_name(identifier):
        # Already an object name, optionally get catalog ID
        try:
            catalog_id = conventions.merianobjectname_to_catalogname(identifier, catalog)
            if catalog_id in catalog.index:
                print(f"📋 Object name: {identifier}")
                print(f"📋 Catalog ID: {catalog_id}")
                return identifier, catalog_id
            else:
                print(f"⚠️  Warning: {catalog_id} not found in catalog")
                return identifier, None
        except Exception as e:
            print(f"⚠️  Could not find catalog ID for {identifier}: {e}")
            return identifier, None

    elif is_merian_id(identifier):
        # Convert Merian ID to object name
        if identifier not in catalog.index:
            raise ValueError(f"Merian ID {identifier} not found in catalog")

        ra = catalog.loc[identifier, 'RA']
        dec = catalog.loc[identifier, 'DEC']
        object_name = conventions.produce_merianobjectname(ra, dec)

        print(f"📋 Catalog ID: {identifier}")
        print(f"📋 Object name: {object_name}")
        print(f"📋 RA, DEC: {ra:.6f}, {dec:.6f}")

        return object_name, identifier

    else:
        raise ValueError(f"Invalid identifier: {identifier}. Must be a Merian ID (M[0-9]+) or object name (J*)")


def check_remote_file(cluster, remote_path):
    """
    Check if a file exists on the remote cluster.

    Parameters
    ----------
    cluster : str
        Cluster hostname
    remote_path : str
        Path to file on remote cluster

    Returns
    -------
    bool
        True if file exists, False otherwise
    """
    cmd = f"ssh {cluster} 'test -f {remote_path} && echo exists || echo missing'"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return 'exists' in result.stdout.lower()
    except subprocess.TimeoutExpired:
        print(f"⚠️  Timeout checking {remote_path}")
        return False
    except Exception as e:
        print(f"⚠️  Error checking {remote_path}: {e}")
        return False


def pull_cutouts_from_path(cluster, dirstem, object_name, output_dir, dry_run=False, verbose=True):
    """
    Pull cutouts from a specific path on the cluster.

    Parameters
    ----------
    cluster : str
        Cluster hostname
    dirstem : str
        Base directory path on cluster
    object_name : str
        Object name (J*)
    output_dir : str
        Local output directory
    dry_run : bool
        If True, print commands but don't execute
    verbose : bool
        Print rsync commands

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    # Create output directories
    merian_dir = os.path.join(output_dir, 'merian')
    hsc_dir = os.path.join(output_dir, 'hsc')

    if not dry_run:
        os.makedirs(merian_dir, exist_ok=True)
        os.makedirs(hsc_dir, exist_ok=True)

    success = True

    # Pull Merian N708 cutouts
    merian_pattern = f"{dirstem}/merian/{object_name}_N708_merim.fits"
    merian_cmd = f"rsync -avz {cluster}:{merian_pattern} {merian_dir}/"

    if verbose or dry_run:
        print(f"🔄 {merian_cmd}")

    if not dry_run:
        result = subprocess.run(merian_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️  Failed to pull Merian cutouts from {dirstem}")
            success = False
        else:
            print(f"✅ Pulled Merian cutouts to {merian_dir}/")

    # Pull HSC cutouts (all bands including g and r) and PSF files
    hsc_pattern = f"{dirstem}/hsc/{object_name}_HSC-*.fits"
    hsc_cmd = f"rsync -avz {cluster}:{hsc_pattern} {hsc_dir}/"

    if verbose or dry_run:
        print(f"🔄 {hsc_cmd}")

    if not dry_run:
        result = subprocess.run(hsc_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️  Failed to pull HSC cutouts from {dirstem}")
            success = False
        else:
            print(f"✅ Pulled HSC cutouts (g, r, i, z, y, etc.) to {hsc_dir}/")

    return success


def pull_cutouts(identifier, output_dir=None, catalog_file=None, config=None,
                 dry_run=False, verbose=True):
    """
    Pull cutouts for a given identifier from the cluster.

    Parameters
    ----------
    identifier : str
        Either a Merian ID (M[0-9]+) or object name (J*)
    output_dir : str
        Local output directory
    catalog_file : str or None
        Path to catalog file. If None, uses value from config.
    config : dict or None
        Configuration dict. If None, loads from default config file.
    dry_run : bool
        If True, print commands but don't execute
    verbose : bool
        Print detailed information
    """
    # Load config if not provided
    if config is None:
        try:
            config = load_config()
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")
            print("Using fallback defaults...")
            config = {}

    # Set defaults from config
    if catalog_file is None:
        catalog_file = config.get('catalog', {}).get('catalog_file')

    if output_dir is None:
        # Default from config
        cutout_config = config.get('cutout_data', {})
        output_dir = cutout_config.get('local_path', '../local_data/cutouts/')

    # Get cluster info from config
    cluster_config = config.get('cutout_data', {}).get('cluster', {})
    cluster = cluster_config.get('hostname', 'tiger3-sumire.princeton.edu')
    remote_paths = cluster_config.get('remote_paths', [
        '/home/kadofong/merian/pieridae/local_data/starbursts_v0',
        '/home/kadofong/merian/pieridae/local_data/msorabove_v0'
    ])

    # Load catalog
    print("📚 Loading catalog...")
    try:
        if catalog_file:
            catalog, masks = sample.load_sample(filename=catalog_file)
        else:
            raise FileNotFoundError("No catalog file specified in config or arguments")
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        return False

    print(f"✅ Loaded catalog with {len(catalog)} objects")

    # Convert identifier to object name
    try:
        object_name, catalog_id = get_object_name(identifier, catalog)
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    print(f"\n🔍 Checking {cluster}...")

    # Try each remote path in order
    for idx, dirstem in enumerate(remote_paths):
        if idx == 0:
            print(f"📂 Trying {Path(dirstem).name}...")
        else:
            print(f"⚠️  Not found, trying {Path(dirstem).name}...")

        # Check if file exists
        test_file = f"{dirstem}/merian/{object_name}_N708_merim.fits"
        if verbose:
            print(f"🔍 Checking for: {test_file}")

        file_exists = check_remote_file(cluster, test_file)

        if file_exists:
            print(f"✅ Found in {Path(dirstem).name}")
            success = pull_cutouts_from_path(
                cluster, dirstem, object_name, output_dir,
                dry_run=dry_run, verbose=verbose
            )

            if success:
                print(f"\n✅ Successfully pulled cutouts to {output_dir}")
                return True
            else:
                print(f"\n❌ Failed to pull cutouts")
                return False

    # If we get here, file wasn't found in any location
    print(f"❌ Cutouts not found in any configured directory")
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Fetch cutouts from tiger3-sumire cluster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s M123456                    # Pull cutouts for Merian ID
  %(prog)s J095618.67+030835.28       # Pull cutouts for object name
  %(prog)s M123456 -o ./my_data/      # Specify output directory
  %(prog)s M123456 --dry-run          # Show commands without executing
        """
    )

    parser.add_argument(
        'identifier',
        type=str,
        help='Merian ID (M[0-9]+) or object name (J*)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for cutouts (default: from config)'
    )

    parser.add_argument(
        '-c', '--catalog-file',
        type=str,
        default=None,
        help='Path to catalog file (default: from config)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file (default: ../configs/figures_config.yaml)'
    )

    parser.add_argument(
        '-d', '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Reduce verbosity'
    )

    args = parser.parse_args()

    # Load config
    config = None
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"⚠️  Error loading config from {args.config}: {e}")
            sys.exit(1)

    success = pull_cutouts(
        args.identifier,
        output_dir=args.output_dir,
        catalog_file=args.catalog_file,
        config=config,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
