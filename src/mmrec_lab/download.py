"""Download the author's official Baby files with resumable, checked transfers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import gdown
import numpy as np


OFFICIAL_FOLDER = "https://drive.google.com/drive/folders/1Fk21441EO1l7wgOOARh2thu4FjgtKWQp"
BABY_FILES = {
    "baby.inter": ("1i2IB2bdxu_jMSxr2IvZ04MG54ySgI2JN", 4_362_239),
    "image_feat.npy": ("103Pxo73naIwmKI6CR71d3t7A5k-lpk9c", 231_014_528),
    "text_feat.npy": ("1EP-Ro9Lq-RQV_Urrh8Xpa-mvr635tytj", 10_828_928),
    "i_id_mapping.csv": ("1eagqR_X3H6zFjw1lFXjLLX7fB7AoTxtG", 111_702),
    "u_id_mapping.csv": ("1IujVzYfihippYT36v16aNLPpKT4ELwyK", 392_164),
}
_NPY_FORMATS = {
    "image_feat.npy": ((7050, 4096), np.dtype("<f8")),
    "text_feat.npy": ((7050, 384), np.dtype("<f4")),
}
_TSV_HEADERS = {
    "baby.inter": ["userID", "itemID", "rating", "timestamp", "x_label"],
    "i_id_mapping.csv": ["asin", "itemID"],
    "u_id_mapping.csv": ["user_id", "userID"],
}


def _validate(path: Path, expected_size: int) -> dict:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path.name}: expected {expected_size} bytes, received {actual_size}")
    details = {"size_bytes": actual_size}
    if path.name in _NPY_FORMATS:
        expected_shape, expected_dtype = _NPY_FORMATS[path.name]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != expected_shape or array.dtype != expected_dtype:
            raise ValueError(f"{path.name}: unexpected NPY shape or dtype")
        details.update(shape=list(array.shape), dtype=str(array.dtype))
        del array  # release Windows mmap before any rename
    else:
        with path.open("r", encoding="utf-8") as stream:
            header = stream.readline().strip().split("\t")
        if header != _TSV_HEADERS[path.name]:
            raise ValueError(f"{path.name}: unexpected TSV header, possibly a download error page")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    details["sha256"] = digest.hexdigest()
    return details


def _preserve_incomplete_final(path: Path) -> None:
    """Keep an accidentally published short file resumable, never as complete."""
    existing = list(path.parent.glob(path.name + "*.part"))
    if existing:
        raise RuntimeError(f"{path.name}: incomplete final file and existing .part files; resolve ambiguity before resuming")
    path.replace(path.with_name(path.name + ".part"))


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _download_file(destination: Path, file_id: str, expected_size: int) -> dict:
    if destination.exists() and destination.stat().st_size != expected_size:
        _preserve_incomplete_final(destination)
    if not destination.exists():
        try:
            result = gdown.download(id=file_id, output=str(destination), resume=True,
                                    use_cookies=False, retries=2, timeout=(20, 60))
        except Exception as exc:
            if destination.exists() and destination.stat().st_size != expected_size:
                _preserve_incomplete_final(destination)
            raise RuntimeError(f"Official Baby download failed for {destination.name}; partial progress is preserved") from exc
        if result is None or not destination.is_file():
            if destination.exists() and destination.stat().st_size != expected_size:
                _preserve_incomplete_final(destination)
            raise RuntimeError(f"Official Baby download did not finish: {destination.name}")
    try:
        return _validate(destination, expected_size)
    except (ValueError, OSError, EOFError) as exc:
        if destination.exists():
            if destination.stat().st_size != expected_size:
                _preserve_incomplete_final(destination)
            else:
                # Correct byte count with a bad header cannot safely be resumed.
                destination.replace(destination.with_name(destination.name + ".invalid." + uuid.uuid4().hex[:8]))
        raise RuntimeError(f"Downloaded Baby file failed validation: {destination.name}") from exc


def download_baby(out_dir: str | Path) -> dict:
    """Return and atomically save a manifest only after all five files validate.

    Official sizes were read from the author's Drive directory. Their SHA256
    hashes are locally calculated, not claimed to be author-published checksums.
    Existing complete files are validated and skipped. Transfers use no account
    cookies and retain gdown's .part files for resume. Two retries mean at most
    three attempts; exhaustion raises an exception for a nonzero CLI exit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "baby", "source": OFFICIAL_FOLDER,
                "complete": False, "checksum_origin": "locally computed SHA256",
                "files": {}}
    manifest_path = out_dir / "download_manifest.json"
    for name, (file_id, expected_size) in BABY_FILES.items():
        destination = out_dir / name
        try:
            details = _download_file(destination, file_id, expected_size)
        except Exception:
            # Replace any old success marker so an interrupted revalidation
            # cannot leave stale metadata claiming these files are complete.
            manifest["failed_file"] = name
            _write_manifest(manifest_path, manifest)
            raise
        details.update(file_id=file_id,
                       url=f"https://drive.google.com/uc?export=download&id={file_id}")
        manifest["files"][name] = details
    manifest["complete"] = True
    _write_manifest(manifest_path, manifest)
    return manifest
