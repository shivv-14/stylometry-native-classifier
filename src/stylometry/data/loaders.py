"""Corpus loaders. Each returns a DataFrame with at least text, label, writer_id."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["text", "label", "writer_id"]
OPTIONAL_COLUMNS = ["topic", "l1", "proficiency"]

DATA_INSTRUCTIONS = """
No corpus found in data/raw/.

This project needs a licensed learner corpus. Nothing is downloaded automatically
and no fake data is generated.

Option A (recommended) - ICNALE Written Essays
  1. Register and download at https://language.sakura.ne.jp/icnale/
     (ICNALE Written Essays, the plain-text version).
  2. Put all essay .txt files, including the ENS (native speaker) files, in
       data/raw/icnale/
     e.g. data/raw/icnale/W_CHN_PTJ0_004_B1_1.txt
  3. Check that data.icnale_filename_regex in configs/config.yaml matches your
     file names, then run: python tasks.py prepare

Option B - LOCNESS as the native class (topic/genre confounding is higher)
  1. Request LOCNESS from UCLouvain (https://uclouvain.be/en/research-institutes/ilc/cecl/locness.html)
  2. Put the LOCNESS .txt files in data/raw/locness/ and the ICNALE files in data/raw/icnale/
  3. Set data.source: icnale+locness in configs/config.yaml

Option C - any corpus as CSV
  Put a CSV at data/raw/dataset.csv with columns text,label,writer_id
  (optional: topic,l1,proficiency; label must be native or non_native) and
  set data.source: csv
"""


class DataNotFoundError(FileNotFoundError):
    pass


def _read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def parse_icnale_filename(name: str, pattern: str, native_code: str = "ENS") -> dict:
    """Parse one ICNALE file name into metadata. Raises ValueError if it does not match."""
    m = re.match(pattern, name)
    if not m:
        raise ValueError(
            f"ICNALE file name '{name}' does not match data.icnale_filename_regex "
            f"({pattern}). Look at the names in data/raw/icnale/ and update the regex "
            "in configs/config.yaml; it needs the named groups country, topic, writer, proficiency."
        )
    g = m.groupdict()
    country = g["country"]
    return {
        "writer_id": f"{country}_{g['writer']}",   # IDs repeat across countries
        "l1": country,
        "topic": g["topic"],
        "proficiency": g.get("proficiency"),
        "label": "native" if country == native_code else "non_native",
    }


def load_icnale(directory: str | Path, pattern: str, native_code: str = "ENS") -> pd.DataFrame:
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*.txt"))
    if not files:
        raise DataNotFoundError(f"No .txt files in {directory}")
    rows, bad = [], []
    for f in files:
        try:
            meta = parse_icnale_filename(f.name, pattern, native_code)
        except ValueError:
            bad.append(f.name)
            continue
        rows.append({"text": _read_text(f), "doc_id": f.stem, "source": "icnale", **meta})
    if bad:
        examples = ", ".join(bad[:5])
        raise ValueError(
            f"{len(bad)} of {len(files)} ICNALE files do not match data.icnale_filename_regex "
            f"(examples: {examples}). Update the regex in configs/config.yaml."
        )
    return pd.DataFrame(rows)


def load_locness(directory: str | Path, header_pattern: str) -> pd.DataFrame:
    """LOCNESS files hold several essays, each starting with a header like <ALEV1001.1>.

    LOCNESS has no reliable writer IDs, so each essay is treated as its own writer
    (documented as a limitation).
    """
    directory = Path(directory)
    files = sorted(directory.rglob("*.txt"))
    if not files:
        raise DataNotFoundError(f"No .txt files in {directory}")
    header = re.compile(header_pattern)
    rows = []
    for f in files:
        current_id, buf = None, []

        def flush():
            text = "\n".join(buf).strip()
            if current_id and text:
                rows.append({
                    "text": text, "doc_id": f"{f.stem}:{current_id}",
                    "writer_id": f"LOC_{f.stem}_{current_id}", "label": "native",
                    "l1": "ENG", "topic": f.stem, "proficiency": "native", "source": "locness",
                })

        for line in _read_text(f).splitlines():
            m = header.match(line)
            if m:
                flush()
                current_id, buf = m.group("id"), []
            else:
                buf.append(line)
        flush()
    if not rows:
        raise ValueError(f"No essays found in {directory}; check data.locness_header_regex.")
    return pd.DataFrame(rows)


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    bad = set(df["label"]) - {"native", "non_native"}
    if bad:
        raise ValueError(f"label must be 'native' or 'non_native', found {sorted(bad)}")
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df["writer_id"] = df["writer_id"].astype(str)
    if "doc_id" not in df.columns:
        df["doc_id"] = [f"doc{i}" for i in range(len(df))]
    df["source"] = "csv"
    return df


def raw_data_available(cfg: dict, root: Path) -> bool:
    raw = root / cfg["paths"]["raw_dir"]
    return raw.exists() and any(p.is_file() and p.name != ".gitkeep" for p in raw.rglob("*"))


def load_raw(cfg: dict, root: Path) -> pd.DataFrame:
    """Load whatever source configs/config.yaml asks for."""
    d, p = cfg["data"], cfg["paths"]
    source = d["source"]
    if source == "csv":
        return load_csv(root / p["csv_path"])
    df = load_icnale(root / p["icnale_dir"], d["icnale_filename_regex"], d["native_country_code"])
    if source == "icnale":
        return df
    if source == "icnale+locness":
        loc = load_locness(root / p["locness_dir"], d["locness_header_regex"])
        # LOCNESS replaces the ICNALE native group
        return pd.concat([df[df.label == "non_native"], loc], ignore_index=True)
    raise ValueError(f"Unknown data.source '{source}'")
