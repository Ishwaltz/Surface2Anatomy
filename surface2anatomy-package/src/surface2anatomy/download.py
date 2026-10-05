"""
Pretrained Weight Caching and Cryptographic Checksum Verification.
Supports platformdirs standard caching, local fallback discovery, and SHA256 integrity checks.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Union, Dict, Any
import platformdirs
import requests
from tqdm import tqdm

from surface2anatomy.exceptions import (
    ModelDownloadError, ChecksumError, ModelNotFoundError
)
from surface2anatomy.manifest import MODEL_MANIFEST

def get_default_cache_dir() -> Path:
    """Returns standard OS user cache directory: ~/.cache/surface2anatomy on Linux."""
    return Path(platformdirs.user_cache_dir("surface2anatomy"))

def compute_sha256(filepath: Union[str, Path]) -> str:
    """Computes hexadecimal SHA256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def verify_file_checksum(filepath: Union[str, Path], expected_sha256: str) -> bool:
    """Verifies that a file matches the expected SHA256 hash. Deletes file if corrupted."""
    actual = compute_sha256(filepath)
    if actual != expected_sha256:
        try:
            os.remove(filepath)
        except OSError:
            pass
        raise ChecksumError(Path(filepath).name, expected_sha256, actual)
    return True

def find_local_artifact(filename: str, expected_sha256: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Searches for an artifact in:
    1. cache_dir or default cache dir
    2. Local workspace repository paths (experiments/, hf_space/checkpoints/)
    """
    search_dirs = []
    if cache_dir is not None:
        search_dirs.append(Path(cache_dir))
    search_dirs.append(get_default_cache_dir())

    # Check local repository locations
    cwd = Path.cwd()
    search_dirs.extend([
        cwd / "experiments" / "phase10R" / "checkpoints",
        cwd / "experiments" / "phase16_brain" / "checkpoints",
        cwd / "hf_space" / "checkpoints",
        cwd.parent / "experiments" / "phase10R" / "checkpoints",
        cwd.parent / "experiments" / "phase16_brain" / "checkpoints",
        cwd.parent / "hf_space" / "checkpoints"
    ])

    for d in search_dirs:
        candidate = d / filename
        if candidate.is_file():
            actual = compute_sha256(candidate)
            if actual == expected_sha256:
                return candidate
    return None

def download_file(urls: list, dest_path: Path, expected_sha256: str) -> Path:
    """Downloads an artifact from prioritized URLs with progress bar and SHA256 verification."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    last_error = None
    for url in urls:
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                with open(temp_path, "wb") as f, tqdm(
                    desc=dest_path.name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    disable=total_size == 0
                ) as bar:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
                
                # Verify checksum before renaming
                verify_file_checksum(temp_path, expected_sha256)
                temp_path.rename(dest_path)
                return dest_path
        except ChecksumError:
            raise
        except Exception as e:
            last_error = e
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            continue

    raise ModelDownloadError(
        f"Failed to download {dest_path.name} from remote mirrors. Last error: {last_error}"
    )

def ensure_model_artifact(
    artifact_meta: Dict[str, Any],
    cache_dir: Optional[Union[str, Path]] = None,
    local_files_only: bool = False
) -> Path:
    """Ensures that a required model artifact is present and cryptographically verified."""
    cache_path = Path(cache_dir) if cache_dir else get_default_cache_dir()
    filename = artifact_meta["filename"]
    expected_sha = artifact_meta["sha256"]

    # 1. Search existing local file
    local_match = find_local_artifact(filename, expected_sha, cache_path)
    if local_match:
        return local_match

    # 2. Check if offline mode prevents downloading
    if local_files_only:
        raise ModelNotFoundError(
            f"Pretrained artifact '{filename}' not found in local cache ({cache_path}) "
            "and local_files_only=True prevents remote downloading."
        )

    # 3. Download to cache directory
    dest = cache_path / filename
    return download_file(artifact_meta["urls"], dest, expected_sha)
