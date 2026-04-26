#!/usr/bin/env python3
"""
Download the DemMA-Planner-SFT checkpoint from HuggingFace Hub.

Default destination
-------------------
    caregiver-r1/checkpoints/demma_planner_sft/

After this runs you should see (≈16 GB total):

    checkpoints/demma_planner_sft/
    ├── action_classifier.pt              ← MLP head over hidden states (NUM_LABELS=34)
    ├── added_tokens.json
    ├── chat_template.jinja
    ├── config.json
    ├── dema_sft_train.py                 ← reference; not needed at inference time
    ├── demma_planner_inference.py        ← reference; minimal inference (no memory)
    ├── generation_config.json
    ├── inference_with_mem.py             ← reference; full inference (with long/short memory)
    ├── memories_27_long.json             ← 27-patient long-term memory profiles
    ├── merges.txt
    ├── metrics.json
    ├── model-00001-of-00004.safetensors  ← Qwen3-8B BF16 shard 1/4
    ├── model-00002-of-00004.safetensors
    ├── model-00003-of-00004.safetensors
    ├── model-00004-of-00004.safetensors
    ├── model.safetensors.index.json
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.json
    └── weekly_schedule.jsonl             ← 27-patient short-term (daily routine) memory

Usage
-----
    cd caregiver-r1
    python scripts/download_demma.py                          # default: ./checkpoints/demma_planner_sft
    python scripts/download_demma.py --dest /scratch/demma    # custom dest
    python scripts/download_demma.py --no-resume              # full re-download

Requires
--------
    pip install huggingface_hub                              # pulled in by requirements-demma.txt

Notes
-----
- The DemMA repo is public and not gated; no HF token required.
- snapshot_download is resumable by default; safe to re-run after a partial download.
- Total download size ≈ 16.4 GB (Qwen3-8B BF16 sharded). Allow ~10 min on a fast link.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "hulehule/DemMA-Planner-SFT"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "checkpoints" / "demma_planner_sft"

EXPECTED_FILES = [
    "action_classifier.pt",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "memories_27_long.json",
    "weekly_schedule.jsonl",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"destination directory (default: {DEFAULT_DEST})")
    parser.add_argument("--no-resume", action="store_true",
                        help="force full re-download (ignore local cache)")
    parser.add_argument("--token", type=str, default=None,
                        help="HF token (only needed if the repo becomes gated; not required as of 2025-11)")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run:", file=sys.stderr)
        print("    pip install -r requirements-demma.txt", file=sys.stderr)
        return 1

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {REPO_ID} → {dest}")
    print(f"    (size ≈ 16.4 GB, may take 5–15 min on a fast link)")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest),
        resume_download=not args.no_resume,
        token=args.token,
        # Don't waste bandwidth on the .gitattributes / pickle preview metadata.
        ignore_patterns=[".gitattributes"],
    )

    print(f"\n✅ Download complete. Verifying expected files...")
    missing: list[str] = []
    for fname in EXPECTED_FILES:
        path = dest / fname
        if not path.exists():
            missing.append(fname)
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"    ✓ {fname}  ({size_mb:.1f} MB)")

    if missing:
        print(f"\n⚠️  Missing expected files: {missing}", file=sys.stderr)
        print(f"    Re-run with --no-resume if the download was interrupted.", file=sys.stderr)
        return 2

    print(f"\n✅ All {len(EXPECTED_FILES)} required files present at {dest}")
    print(f"\nNext step: configure caregiver-r1/configs/smoke_run.yaml:")
    print(f"    demma:")
    print(f"      mode: real")
    print(f"      real:")
    print(f"        model_path: {dest}")
    print(f"        classifier_path: {dest / 'action_classifier.pt'}")
    print(f"        long_memory_path: {dest / 'memories_27_long.json'}")
    print(f"        short_memory_path: {dest / 'weekly_schedule.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
