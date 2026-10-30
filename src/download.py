from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/nicholasjhana/energy-consumption-generation-prices-and-weather"
PAGE = "https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather"
FILES = {"energy_dataset.csv": b"time,", "weather_features.csv": b"dt_iso,"}
EXPECTED_ARCHIVE_SHA256 = "5448bc8906e5bc73a7c1b9ab92eadab82508b4c350cec30419a5745cda848c31"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_data(root: Path = ROOT, force: bool = False) -> dict:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = raw / "download_manifest.json"
    if not force and manifest.exists():
        info = json.loads(manifest.read_text())
        if all((raw / name).exists() and sha256(raw / name) == item["sha256"]
               for name, item in info["files"].items()):
            return info
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "renewable-risk-analysis/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read(30_000_001)
                status = response.status
                content_type = response.headers.get("Content-Type", "")
            if len(body) > 30_000_000:
                raise ValueError("Archive exceeds the expected size limit.")
            digest = hashlib.sha256(body).hexdigest()
            if status != 200 or not zipfile.is_zipfile(io.BytesIO(body)):
                raise ValueError("The URL did not return a valid ZIP archive.")
            if digest != EXPECTED_ARCHIVE_SHA256:
                raise ValueError("The original dataset version has changed; review it before updating the checksum.")
            contents = {}
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                for name, prefix in FILES.items():
                    record = archive.getinfo(name)
                    if record.file_size > 35_000_000:
                        raise ValueError(f"Unexpected uncompressed size: {name}")
                    data = archive.read(name)
                    if not data.startswith(prefix):
                        raise ValueError(f"Unexpected CSV header: {name}")
                    contents[name] = data
            for name, data in contents.items():
                temporary = raw / (name + ".part")
                temporary.write_bytes(data)
                temporary.replace(raw / name)
            info = {
                "dataset_page": PAGE, "download_url": URL,
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": status, "content_type": content_type,
                "archive_bytes": len(body), "archive_sha256": digest,
                "license": "CC0: Public Domain (Kaggle dataset metadata)",
                "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in contents.items()},
            }
            manifest.write_text(json.dumps(info, indent=2) + "\n")
            return info
        except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Download failed after three attempts: {error}") from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(download_data(force=arguments.force), indent=2))
