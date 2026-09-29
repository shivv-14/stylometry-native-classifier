"""Download the W&I+LOCNESS v2.1 corpus (BEA-2019 shared task) into data/raw/wi_locness/.

    python scripts/download_data.py

The corpus is free to download for non-commercial research and education. By
downloading it you accept the licences in data/raw/wi_locness/licence.wi.txt and
license.locness.txt. It is gitignored and must not be redistributed.
"""
from __future__ import annotations

import io
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stylometry.config import load_config, rel, resolve  # noqa: E402

URL = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"


def main() -> int:
    cfg = load_config()
    out = resolve(cfg, "wi_locness_dir")
    if any(out.glob("json/*.json")):
        print(f"already downloaded: {rel(out)}")
        return 0
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r:
        data = r.read()
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            name = m.name.split("/", 1)[1] if "/" in m.name else ""
            # keep essays + licences only (skip M2 annotation files and the unlabelled test set)
            if not m.isfile() or not (name.startswith("json/") or name.startswith("licen") or name == "readme.txt"):
                continue
            target = (out / name).resolve()
            if out.resolve() not in target.parents:
                raise ValueError(f"unsafe path in archive: {m.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(m).read())
    print(f"saved to {rel(out)}. Licences: licence.wi.txt, license.locness.txt (non-commercial use only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
