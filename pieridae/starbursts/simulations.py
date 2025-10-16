"""
Tools for fetching and managing FIRE2 simulation data.

This module provides utilities to download FIRE2 simulation data from the public
release at https://users.flatironinstitute.org/~mgrudic/fire2_public_release/
and manage local cached copies with automatic cleanup of stale files.
"""

import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List
import warnings
import re
from html.parser import HTMLParser


class _DirectoryListingParser(HTMLParser):
    """Parse Apache-style directory listing HTML to extract file and directory names."""

    def __init__(self):
        super().__init__()
        self.items = []
        self._in_link = False
        self._current_href = None

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self._in_link = True
            for attr, value in attrs:
                if attr == 'href':
                    self._current_href = value

    def handle_endtag(self, tag):
        if tag == 'a':
            self._in_link = False
            self._current_href = None

    def handle_data(self, data):
        if self._in_link and self._current_href:
            # Skip parent directory link
            if self._current_href == '../':
                return
            # Only keep hrefs that match the link text (filters out irrelevant links)
            if data.strip() and self._current_href.startswith(data.strip()):
                self.items.append(self._current_href)


def _get_directory_listing(url: str) -> tuple[List[str], List[str]]:
    """
    Fetch and parse a directory listing from a URL.

    Parameters
    ----------
    url : str
        URL to the directory

    Returns
    -------
    tuple[List[str], List[str]]
        (list of subdirectories, list of files)
    """
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read().decode('utf-8')

        parser = _DirectoryListingParser()
        parser.feed(html)

        # Separate directories (ending with /) from files
        directories = [item.rstrip('/') for item in parser.items if item.endswith('/')]
        files = [item for item in parser.items if not item.endswith('/')]

        return directories, files

    except Exception as e:
        warnings.warn(f"Failed to parse directory listing from {url}: {e}")
        return [], []


def _cleanup_old_files(base_directory: str, max_age_days: int = 7) -> tuple[int, int]:
    """
    Remove files that haven't been accessed in the specified number of days.

    Traverses the directory tree and deletes files (not directories) based on
    their last access time. Directories themselves are never deleted.

    Parameters
    ----------
    base_directory : str
        Root directory to start the cleanup traversal
    max_age_days : int, optional
        Maximum age in days for files to keep (default: 7)

    Returns
    -------
    tuple[int, int]
        (number of files deleted, total size freed in bytes)
    """
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 3600  # Convert days to seconds
    cutoff_time = current_time - max_age_seconds

    deleted_count = 0
    freed_bytes = 0

    for root, dirs, files in os.walk(base_directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                # Get last access time
                last_access = os.path.getatime(filepath)

                # Check if file is older than cutoff
                if last_access < cutoff_time:
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted_count += 1
                    freed_bytes += file_size
                    print(f"  Removed old file: {filepath} (last accessed {(current_time - last_access) / 86400:.1f} days ago)")

            except (OSError, FileNotFoundError) as e:
                warnings.warn(f"Could not process {filepath}: {e}")
                continue

    return deleted_count, freed_bytes


def fetch_fire2_data(
    relative_path: str,
    base_url: str = "https://users.flatironinstitute.org/~mgrudic/fire2_public_release",
    local_base: Optional[str] = None,
    cleanup_age_days: int = 7,
    force_redownload: bool = False
) -> str:
    """
    Download a file from the FIRE2 public data release.

    This function downloads a specific file from the FIRE2 simulation data repository
    and saves it to the local mirror directory structure. Before downloading, it
    performs cleanup of files that haven't been accessed in the specified time period.

    Parameters
    ----------
    relative_path : str
        Path to the file relative to the FIRE2 base URL.
        Examples:
            - "README.txt"
            - "core/m11i_res7100/snapshot_times.txt"
            - "high_redshift/z5m09a/output/snapshot_000.hdf5"
    base_url : str, optional
        Base URL for the FIRE2 data release
        (default: "https://users.flatironinstitute.org/~mgrudic/fire2_public_release")
    local_base : str, optional
        Local base directory for storing downloaded files.
        If None, defaults to merger_analysis/local_data/fire2/ relative to the
        pieridae package location.
    cleanup_age_days : int, optional
        Delete local files not accessed in this many days before downloading (default: 7).
        Set to 0 to disable cleanup.
    force_redownload : bool, optional
        If True, download the file even if it already exists locally (default: False)

    Returns
    -------
    str
        Absolute path to the downloaded file

    Raises
    ------
    urllib.error.URLError
        If the download fails due to network issues
    ValueError
        If the relative_path is invalid or attempts directory traversal

    Examples
    --------
    >>> # Download a README file
    >>> filepath = fetch_fire2_data("README.txt")

    >>> # Download a specific snapshot
    >>> filepath = fetch_fire2_data("core/m11i_res7100/snapshot_times.txt")

    >>> # Skip cleanup for this download
    >>> filepath = fetch_fire2_data("high_redshift/z5m09a/output/data.hdf5",
    ...                              cleanup_age_days=0)
    """
    # Validate relative path (prevent directory traversal attacks)
    if ".." in relative_path or relative_path.startswith("/"):
        raise ValueError(f"Invalid relative path: {relative_path}")

    # Determine local base directory
    if local_base is None:
        # Default to merger_analysis/local_data/fire2/
        # This assumes the script is in pieridae/starbursts/
        package_dir = Path(__file__).parent.parent.parent  # Go up to pieridae repo root
        local_base = package_dir / "merger_analysis" / "local_data" / "fire2"
    else:
        local_base = Path(local_base)

    # Ensure local base directory exists
    local_base.mkdir(parents=True, exist_ok=True)

    # Perform cleanup of old files before downloading
    if cleanup_age_days > 0:
        print(f"Cleaning up files older than {cleanup_age_days} days in {local_base}...")
        deleted_count, freed_bytes = _cleanup_old_files(str(local_base), cleanup_age_days)
        if deleted_count > 0:
            print(f"  Deleted {deleted_count} file(s), freed {freed_bytes / (1024**2):.2f} MB")
        else:
            print("  No old files to clean up")

    # Construct full URL and local path
    full_url = f"{base_url.rstrip('/')}/{relative_path}"
    local_path = local_base / relative_path

    # Create parent directories if needed
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if local_path.exists() and not force_redownload:
        print(f"File already exists at {local_path}")
        # Update access time to mark as recently used
        os.utime(local_path, None)
        return str(local_path)

    # Download the file
    print(f"Downloading {full_url}...")
    print(f"  -> {local_path}")

    try:
        # Download with progress indication for large files
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                downloaded_mb = downloaded / (1024**2)
                total_mb = total_size / (1024**2)
                print(f"\r  Progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)",
                      end='', flush=True)

        urllib.request.urlretrieve(full_url, local_path, reporthook=report_progress)
        print()  # New line after progress
        print(f"Successfully downloaded to {local_path}")

        return str(local_path)

    except urllib.error.HTTPError as e:
        print(f"\nHTTP Error {e.code}: {e.reason}")
        print(f"URL: {full_url}")
        raise
    except urllib.error.URLError as e:
        print(f"\nURL Error: {e.reason}")
        print(f"URL: {full_url}")
        raise
    except Exception as e:
        print(f"\nUnexpected error during download: {e}")
        # Clean up partial download if it exists
        if local_path.exists():
            local_path.unlink()
        raise


def list_fire2_structure(
    local_base: Optional[str] = None,
    max_depth: int = 2
) -> None:
    """
    Print the local FIRE2 directory structure.

    Parameters
    ----------
    local_base : str, optional
        Local base directory. If None, uses default location.
    max_depth : int, optional
        Maximum depth to display (default: 2)
    """
    if local_base is None:
        package_dir = Path(__file__).parent.parent.parent
        local_base = package_dir / "merger_analysis" / "local_data" / "fire2"
    else:
        local_base = Path(local_base)

    if not local_base.exists():
        print(f"Directory does not exist: {local_base}")
        return

    def print_tree(directory: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return

        contents = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))

        for i, item in enumerate(contents):
            is_last = i == len(contents) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}{'/' if item.is_dir() else ''}")

            if item.is_dir():
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension, depth + 1)

    print(f"{local_base}/")
    print_tree(local_base)


def fetch_fire2_directory(
    relative_path: str = "",
    base_url: str = "https://users.flatironinstitute.org/~mgrudic/fire2_public_release",
    local_base: Optional[str] = None,
    cleanup_age_days: int = 7,
    max_depth: Optional[int] = None,
    file_pattern: Optional[str] = None,
    skip_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Recursively download all files in a directory from the FIRE2 public data release.

    This function crawls the directory structure on the FIRE2 server and downloads
    all files found, maintaining the directory structure locally. Before downloading,
    it performs cleanup of files that haven't been accessed in the specified time period.

    Parameters
    ----------
    relative_path : str, optional
        Path to the directory relative to the FIRE2 base URL (default: "" for root).
        Examples:
            - "" (root directory)
            - "core/m11i_res7100"
            - "high_redshift/z5m09a/output"
    base_url : str, optional
        Base URL for the FIRE2 data release
        (default: "https://users.flatironinstitute.org/~mgrudic/fire2_public_release")
    local_base : str, optional
        Local base directory for storing downloaded files.
        If None, defaults to merger_analysis/local_data/fire2/ relative to the
        pieridae package location.
    cleanup_age_days : int, optional
        Delete local files not accessed in this many days before downloading (default: 7).
        Set to 0 to disable cleanup. Cleanup only runs once at the start.
    max_depth : int, optional
        Maximum depth to recurse into subdirectories. None means unlimited (default: None).
        depth=0 downloads only files in the specified directory
        depth=1 includes one level of subdirectories, etc.
    file_pattern : str, optional
        Regular expression pattern to filter files. Only files matching this pattern
        will be downloaded. None means download all files (default: None).
        Example: r"\.hdf5$" to download only HDF5 files
    skip_patterns : List[str], optional
        List of regular expression patterns for paths to skip.
        Example: [r"^output/", r"snapshot_\d{3}\.hdf5$"]

    Returns
    -------
    List[str]
        List of absolute paths to all downloaded files

    Raises
    ------
    ValueError
        If the relative_path is invalid or attempts directory traversal

    Examples
    --------
    >>> # Download all files in a specific simulation directory
    >>> files = fetch_fire2_directory("core/m11i_res7100")

    >>> # Download only HDF5 files from a directory, max 2 levels deep
    >>> files = fetch_fire2_directory("high_redshift/z5m09a",
    ...                                 file_pattern=r"\.hdf5$",
    ...                                 max_depth=2)

    >>> # Download root-level files only (no subdirectories)
    >>> files = fetch_fire2_directory("", max_depth=0)

    >>> # Download but skip large output directories
    >>> files = fetch_fire2_directory("core/m11i_res7100",
    ...                                 skip_patterns=[r"output/snapdir_"])
    """
    # Validate relative path
    if ".." in relative_path or relative_path.startswith("/"):
        raise ValueError(f"Invalid relative path: {relative_path}")

    # Determine local base directory
    if local_base is None:
        package_dir = Path(__file__).parent.parent.parent
        local_base = package_dir / "merger_analysis" / "local_data" / "fire2"
    else:
        local_base = Path(local_base)

    # Ensure local base directory exists
    local_base.mkdir(parents=True, exist_ok=True)

    # Perform cleanup once at the start
    if cleanup_age_days > 0:
        print(f"Cleaning up files older than {cleanup_age_days} days in {local_base}...")
        deleted_count, freed_bytes = _cleanup_old_files(str(local_base), cleanup_age_days)
        if deleted_count > 0:
            print(f"  Deleted {deleted_count} file(s), freed {freed_bytes / (1024**2):.2f} MB")
        else:
            print("  No old files to clean up")
        print()

    # Compile patterns
    file_regex = re.compile(file_pattern) if file_pattern else None
    skip_regexes = [re.compile(p) for p in skip_patterns] if skip_patterns else []

    downloaded_files = []
    failed_downloads = []

    def _should_skip(path: str) -> bool:
        """Check if path matches any skip pattern."""
        return any(regex.search(path) for regex in skip_regexes)

    def _download_directory(rel_path: str, depth: int = 0):
        """Recursively download directory contents."""
        # Check depth limit
        if max_depth is not None and depth > max_depth:
            return

        # Check if this path should be skipped
        if _should_skip(rel_path):
            print(f"Skipping {rel_path} (matches skip pattern)")
            return

        # Construct URL for this directory
        if rel_path:
            url = f"{base_url.rstrip('/')}/{rel_path}"
        else:
            url = base_url.rstrip('/')

        print(f"\nExploring: {url}")

        # Get directory listing
        subdirs, files = _get_directory_listing(url)

        print(f"  Found {len(files)} file(s) and {len(subdirs)} subdirectory(ies)")

        # Download files in this directory
        for filename in files:
            file_rel_path = f"{rel_path}/{filename}" if rel_path else filename

            # Check skip patterns
            if _should_skip(file_rel_path):
                print(f"  Skipping {filename} (matches skip pattern)")
                continue

            # Check file pattern
            if file_regex and not file_regex.search(filename):
                print(f"  Skipping {filename} (doesn't match file pattern)")
                continue

            # Download the file
            try:
                filepath = fetch_fire2_data(
                    file_rel_path,
                    base_url=base_url,
                    local_base=str(local_base),
                    cleanup_age_days=0,  # Already cleaned up
                    force_redownload=False
                )
                downloaded_files.append(filepath)
            except Exception as e:
                print(f"  ERROR downloading {filename}: {e}")
                failed_downloads.append(file_rel_path)

        # Recurse into subdirectories
        if max_depth is None or depth < max_depth:
            for dirname in subdirs:
                subdir_rel_path = f"{rel_path}/{dirname}" if rel_path else dirname
                _download_directory(subdir_rel_path, depth + 1)

    # Start the recursive download
    print(f"Starting recursive download from: {base_url.rstrip('/')}/{relative_path}")
    print(f"Saving to: {local_base / relative_path}")
    if max_depth is not None:
        print(f"Maximum depth: {max_depth}")
    if file_pattern:
        print(f"File pattern: {file_pattern}")
    if skip_patterns:
        print(f"Skip patterns: {skip_patterns}")

    _download_directory(relative_path, depth=0)

    # Summary
    print("\n" + "="*70)
    print(f"Download complete!")
    print(f"  Successfully downloaded: {len(downloaded_files)} file(s)")
    if failed_downloads:
        print(f"  Failed downloads: {len(failed_downloads)} file(s)")
        print(f"  Failed files: {failed_downloads[:10]}")  # Show first 10
        if len(failed_downloads) > 10:
            print(f"    ... and {len(failed_downloads) - 10} more")
    print("="*70)

    return downloaded_files
