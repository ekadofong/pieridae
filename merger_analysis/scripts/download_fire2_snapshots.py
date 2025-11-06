#!/usr/bin/env python3
"""
Download specific FIRE2 snapshots for analysis.

This script downloads snapshots 400, 500, and 600 for simulations:
- m11d_res7100
- m11h_res7100
- m11b_res2100
"""

import sys
from pathlib import Path

# Add pieridae to path
pieridae_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(pieridae_path))

from pieridae.starbursts.simulations import fetch_fire2_data


def main():
    """Download specified FIRE2 snapshots."""

    # Define simulations and snapshots to download
    simulations = [
        "m11d_res7100",
        "m11h_res7100",
        "m11b_res2100"
    ]

    snapshots = [400, 500, 600]

    # Track downloads
    downloaded_files = []
    failed_downloads = []

    print("="*70)
    print("FIRE2 Snapshot Download")
    print("="*70)
    print(f"Simulations: {', '.join(simulations)}")
    print(f"Snapshots: {', '.join(map(str, snapshots))}")
    print("="*70)
    print()

    # Download each combination
    for sim in simulations:
        for snap in snapshots:
            # Construct path (assuming snapshots are in core/[sim]/output/)
            snapshot_filename = f"snapshot_{snap:03d}.hdf5"
            relative_path = f"core/{sim}/output/{snapshot_filename}"

            print(f"\n{'='*70}")
            print(f"Downloading: {sim} - snapshot {snap}")
            print(f"Path: {relative_path}")
            print(f"{'='*70}")

            try:
                filepath = fetch_fire2_data(
                    relative_path,
                    cleanup_age_days=7  # Clean up files older than 7 days
                )
                downloaded_files.append(filepath)
                print(f"✓ Successfully downloaded to: {filepath}")

            except Exception as e:
                print(f"✗ Failed to download {relative_path}")
                print(f"  Error: {e}")
                failed_downloads.append(relative_path)

    # Print summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"Successfully downloaded: {len(downloaded_files)} file(s)")
    print(f"Failed downloads: {len(failed_downloads)} file(s)")

    if downloaded_files:
        print("\nSuccessful downloads:")
        for filepath in downloaded_files:
            print(f"  ✓ {filepath}")

    if failed_downloads:
        print("\nFailed downloads:")
        for path in failed_downloads:
            print(f"  ✗ {path}")

    print("="*70)

    return len(failed_downloads) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
