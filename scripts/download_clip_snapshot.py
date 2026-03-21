# -*- coding: utf-8 -*-
"""
Download CLIP model snapshot to a local folder.

Fallback when `huggingface-cli` command is not available.
"""

import os
from pathlib import Path


def main():
    repo_id = "openai/clip-vit-base-patch32"
    out_dir = Path(r"D:\models\clip-vit-base-patch32")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f"Cannot import huggingface_hub.snapshot_download: {e}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Downloaded {repo_id} to {out_dir}")


if __name__ == "__main__":
    main()

