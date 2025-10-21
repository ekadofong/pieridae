#!/usr/bin/env python
"""
Pull Cutouts from Tiger3-Sumire Cluster

This script pulls cutouts (both Merian and HSC) from the tiger3-sumire.princeton.edu
cluster for a given Merian ID or object name. It tries starbursts_v0 first, then
falls back to msorabove_v0 if files are not found.

Usage:
    python pull_cutouts.py M123456  # Using Merian ID
    python pull_cutouts.py J095618.67+030835.28  # Using object name
    python pull_cutouts.py M123456 -o ./my_data/  # Specify output directory
"""

import os
import sys
import argparse
import subprocess
import re
from pathlib import Path

# Add pieridae to path (go up 2 levels: figure_generation/ -> merger_analysis/ -> pieridae/)
sys.path.insert(0, str(Path(__file__).parents[2]))

from carpenter import conventions
from pieridae.starbursts import sample


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


def pull_cutouts(identifier, output_dir='./figure_data/', catalog_file=None,
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
        Path to catalog file. If None, uses default.
    dry_run : bool
        If True, print commands but don't execute
    verbose : bool
        Print detailed information
    """
    # Load catalog
    print("📚 Loading catalog...")
    try:
        if catalog_file:
            catalog, masks = sample.load_sample(filename=catalog_file)
        else:
            try:
                catalog, masks = sample.load_sample()
            except FileNotFoundError:
                # Try common locations
                possible_paths = [
                    '../../local_data/base_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet',
                    '../local_data/base_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet',
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        catalog, masks = sample.load_sample(filename=path)
                        break
                else:
                    raise FileNotFoundError(
                        "Could not find catalog file. Please specify with --catalog-file"
                    )
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

    # Cluster information
    cluster = 'tiger3-sumire.princeton.edu'

    # Try starbursts_v0 first
    dirstem_starbursts = '/home/kadofong/merian/pieridae/local_data/starbursts_v0'
    dirstem_msorabove = '/home/kadofong/merian/pieridae/local_data/msorabove_v0'

    print(f"\n🔍 Checking {cluster}...")
    print(f"📂 Trying starbursts_v0 first...")

    # Check if file exists in starbursts_v0
    test_file = f"{dirstem_starbursts}/merian/{object_name}_N708_merim.fits"
    if verbose:
        print(f"🔍 Checking for: {test_file}")

    file_exists = check_remote_file(cluster, test_file)

    if file_exists:
        print(f"✅ Found in starbursts_v0")
        success = pull_cutouts_from_path(
            cluster, dirstem_starbursts, object_name, output_dir,
            dry_run=dry_run, verbose=verbose
        )
    else:
        print(f"⚠️  Not found in starbursts_v0, trying msorabove_v0...")
        test_file = f"{dirstem_msorabove}/merian/{object_name}_N708_merim.fits"
        if verbose:
            print(f"🔍 Checking for: {test_file}")

        file_exists = check_remote_file(cluster, test_file)

        if file_exists:
            print(f"✅ Found in msorabove_v0")
            success = pull_cutouts_from_path(
                cluster, dirstem_msorabove, object_name, output_dir,
                dry_run=dry_run, verbose=verbose
            )
        else:
            print(f"❌ Cutouts not found in either directory")
            return False

    if success:
        print(f"\n✅ Successfully pulled cutouts to {output_dir}")
        return True
    else:
        print(f"\n❌ Failed to pull cutouts")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Pull cutouts from tiger3-sumire cluster',
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
        default='./figure_data/',
        help='Output directory for cutouts (default: ./figure_data/)'
    )

    parser.add_argument(
        '-c', '--catalog-file',
        type=str,
        default='/Users/kadofong/work/projects/merian/local_data/base_catalogs/mdr1_n708maglt26_and_pzgteq0p1.parquet',
        help='Path to catalog file (optional)'
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

    success = pull_cutouts(
        args.identifier,
        output_dir=args.output_dir,
        catalog_file=args.catalog_file,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
