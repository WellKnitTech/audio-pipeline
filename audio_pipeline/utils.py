from __future__ import annotations

import logging
import re
import shlex
import subprocess
from pathlib import Path
from typing import List

def run_cmd(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    logging.debug("CMD: %s", " ".join(shlex.quote(c) for c in cmd))
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if check and cp.returncode != 0:
        raise RuntimeError(f"Command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stdout}")
    return cp

def ensure_ffmpeg() -> str:
    cp = run_cmd(["bash", "-lc", "command -v ffmpeg"], check=False)
    path = cp.stdout.strip()
    if cp.returncode != 0 or not path:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg and retry.")
    return path

def ffmpeg_filters() -> str:
    cp = run_cmd(["ffmpeg", "-hide_banner", "-filters"], check=True)
    return cp.stdout

def has_filter(filters_text: str, name: str) -> bool:
    return re.search(rf"(^|[\s]){re.escape(name)}([\s]|$)", filters_text, flags=re.MULTILINE) is not None

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
