"""Extract the parts of Agriculture-Vision 2021 (supervised) that CropWatch uses.

Reads the public ~21 GB archive and writes, for train and val:
    images/rgb, masks, boundaries, labels/<class>
NIR images and the unlabelled test split are skipped. Nothing is extracted from archive
paths directly: output paths are built from validated names only.

By downloading you agree to the Agriculture-Vision Workshop Terms and Conditions
(non-commercial use): https://github.com/SHI-Labs/Agriculture-Vision

Recommended (two steps, run from the repo root in your own terminal):
    1. Download the archive; -C - resumes if it is interrupted (~2 hours at ~3 MB/s):
       curl.exe -L -C - -o ml\\data\\Agriculture-Vision-2021.tar.gz <URL below>
    2. Extract what we need from the local file:
       ml\\.venv\\Scripts\\python.exe ml\\scripts\\download_agvision.py ^
           --source ml\\data\\Agriculture-Vision-2021.tar.gz
Without --source the script streams straight from the URL (no resume: if it stops, it
starts again from the beginning).
"""

import argparse
import collections
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

URL = (
    "https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/"
    "cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz"
)
ML_DIR = Path(__file__).resolve().parents[1]
KEY_RE = re.compile(r"^[A-Z0-9]+_\d+-\d+-\d+-\d+$")
NAME_RE = re.compile(r"^[a-z0-9_]+$")
KEEP_SPLITS = {"train", "val"}
KEEP_FOLDERS = {("images", "rgb"), ("masks",), ("boundaries",)}  # plus ("labels", <class>)


class Progress:
    """File-like wrapper that counts bytes and prints progress."""

    def __init__(self, f, every_mb: int, max_bytes: int | None) -> None:
        self.f, self.n, self.max_bytes = f, 0, max_bytes
        self.every, self.next_report, self.start = (
            every_mb * 1024**2,
            every_mb * 1024**2,
            time.time(),
        )

    def read(self, size: int = -1) -> bytes:
        if self.max_bytes is not None and self.n >= self.max_bytes:
            raise EOFError("--max-bytes reached")
        b = self.f.read(size)
        self.n += len(b)
        if self.n >= self.next_report:
            mb, secs = self.n / 1024**2, time.time() - self.start
            print(f"[{mb:7.0f} MB] {mb / secs:4.1f} MB/s, {secs / 60:5.1f} min", flush=True)
            self.next_report += self.every
        return b


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help="local Agriculture-Vision-2021.tar.gz (default: stream from the public URL)",
    )
    p.add_argument("--out", type=Path, default=ML_DIR / "data/agvision2021")
    p.add_argument("--max-bytes", type=int, default=None, help="stop early (for testing)")
    p.add_argument("--report-every-mb", type=int, default=500)
    args = p.parse_args()
    if args.source is not None and not args.source.is_file():
        sys.exit(f"--source not found: {args.source}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "COMPLETE").unlink(missing_ok=True)

    if args.source is not None:
        raw = args.source.open("rb")
        print(f"reading {args.source} ({args.source.stat().st_size:,} bytes)", flush=True)
    else:
        raw = urllib.request.urlopen(URL)
        print(f"streaming {URL}", flush=True)
    src = Progress(raw, args.report_every_mb, args.max_bytes)
    saved: collections.Counter[str] = collections.Counter()
    seen: collections.Counter[str] = collections.Counter()
    finished = False
    try:
        with tarfile.open(fileobj=src, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                parts = m.name.split("/")
                if len(parts) < 4 or parts[0] != "Agriculture-Vision-2021":
                    continue
                split, folder, fname = parts[1], tuple(parts[2:-1]), parts[-1]
                fkey = f"{split}/{'/'.join(folder)}"
                if seen[fkey] == 0:
                    print(f"[{src.n / 1024**2:7.0f} MB] folder {fkey}", flush=True)
                seen[fkey] += 1
                if split not in KEEP_SPLITS:
                    continue
                is_label = len(folder) == 2 and folder[0] == "labels" and NAME_RE.match(folder[1])
                if folder not in KEEP_FOLDERS and not is_label:
                    continue
                stem, _, ext = fname.rpartition(".")
                if not KEY_RE.match(stem) or ext.lower() not in ("jpg", "png"):
                    continue
                dest = out.joinpath(split, *folder, f"{stem}.{ext.lower()}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tf.extractfile(m).read())
                saved[fkey] += 1
        finished = True
    except (EOFError, OSError, tarfile.TarError) as e:
        print(f"(stopped early: {type(e).__name__}: {e})", flush=True)

    print(f"\ndownloaded {src.n / 1024**2:.0f} MB in {(time.time() - src.start) / 60:.1f} min")
    print("files saved per folder:")
    for k in sorted(saved):
        print(f"  {saved[k]:7d}  {k}")
    if finished:
        (out / "COMPLETE").write_text("ok\n")
        print(f"complete: {out}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
