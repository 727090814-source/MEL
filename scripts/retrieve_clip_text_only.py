# -*- coding: utf-8 -*-
"""
CLIP 文本向量-only 检索：mention 文本 embedding 对 KB 实体文本 embedding 全库相似度检索。

使用 embedding_clip/{dataset}/entity_text.pt 与 mention_text.pt（512 维），
L2 归一化后做余弦检索（矩阵乘法），输出与 retrieve_clip_concat 相同结构的
metrics.json / candidate-*.json。

用法:
  python scripts/retrieve_clip_text_only.py --dataset all
  python scripts/retrieve_clip_text_only.py --dataset WikiMEL --k_values 1,5,10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from retrieve_clip_concat import (  # noqa: E402
    build_idx2qid,
    load_json,
    load_splits_ordered,
    load_mention_splits,
    run_retrieval_split,
    verify_alignment,
)


def run_dataset_text(
    dataset: str,
    data_root: Path,
    embed_root: Path,
    out_root: Path,
    num_candidates: int,
    batch_size: int,
    device: torch.device,
    k_values: List[int],
) -> Dict[str, dict]:
    ds_root = data_root / dataset
    emb_dir = embed_root / dataset
    out_dir = out_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    req = [
        emb_dir / "entity_text.pt",
        emb_dir / "mention_text.pt",
        emb_dir / "entity_qid2idx.json",
        emb_dir / "mention_key2idx.json",
    ]
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError(f"[{dataset}] missing:\n" + "\n".join(missing))

    qid2idx: Dict[str, int] = {str(k): int(v) for k, v in load_json(emb_dir / "entity_qid2idx.json").items()}
    mention_key2idx: Dict[str, int] = {
        str(k): int(v) for k, v in load_json(emb_dir / "mention_key2idx.json").items()
    }
    idx2qid = build_idx2qid(qid2idx)

    et = torch.load(emb_dir / "entity_text.pt", map_location="cpu")
    mt = torch.load(emb_dir / "mention_text.pt", map_location="cpu")
    entity_emb = F.normalize(et.float(), p=2, dim=1)
    mention_emb = F.normalize(mt.float(), p=2, dim=1)

    mention_items = load_mention_splits(ds_root)
    if len(mention_items) != mention_emb.shape[0]:
        raise ValueError(
            f"[{dataset}] mention count {len(mention_items)} != tensor rows {mention_emb.shape[0]}"
        )
    if not verify_alignment(mention_items, mention_key2idx):
        print(f"[{dataset}] alignment check failed; results may be wrong.")

    splits = load_splits_ordered(ds_root)
    if not splits:
        raise RuntimeError(f"[{dataset}] could not find train/dev/test json triple under {ds_root}")

    split_key = {"train": "train", "dev": "val", "test": "test"}
    candidate_bundle: Dict[str, dict] = {}
    metrics_all: Dict[str, dict] = {}

    offset = 0
    for split_name, items in splits:
        n = len(items)
        sl_emb = mention_emb[offset : offset + n]
        offset += n
        preds, met = run_retrieval_split(
            mention_emb=sl_emb,
            entity_emb=entity_emb,
            idx2qid=idx2qid,
            items=items,
            qid2idx=qid2idx,
            device=device,
            batch_size=batch_size,
            k_values=k_values,
            num_candidates=num_candidates,
        )
        candidate_bundle[split_key[split_name]] = preds
        metrics_all[split_key[split_name]] = {**met, "split": split_name}

    cand_path = out_dir / f"candidate-{num_candidates}.json"
    with cand_path.open("w", encoding="utf-8") as f:
        json.dump(candidate_bundle, f, ensure_ascii=False, indent=2)
    print(f"[{dataset}] saved {cand_path}")

    met_path = out_dir / "metrics.json"
    with met_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)
    print(f"[{dataset}] saved {met_path}")

    for sk, met in metrics_all.items():
        h = " ".join([f"H@{k}={100 * met.get(f'hits@{k}', 0):.2f}" for k in k_values])
        print(f"[{dataset}] {sk} ({met.get('split')}): {h}  MRR={100 * met['mrr']:.2f}  n={met['n']}")

    return metrics_all


def main():
    parser = argparse.ArgumentParser(description="CLIP text-only retrieval (mention_text vs entity_text)")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--embed_root", type=str, default="embedding_clip")
    parser.add_argument("--out_root", type=str, default="results/clip_retrieve_text")
    parser.add_argument("--dataset", type=str, default="all", help="WikiMEL | WikiDiverse | RichpediaMEL | all")
    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--k_values", type=str, default="1,5,10")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    data_root = Path(args.data_root)
    embed_root = Path(args.embed_root)
    out_root = Path(args.out_root)
    datasets = ["WikiMEL", "WikiDiverse", "RichpediaMEL"] if args.dataset == "all" else [args.dataset]

    print(f"[text-only] device={device}  num_candidates={args.num_candidates}  k_values={k_values}")

    summary: Dict[str, dict] = {}
    for ds in datasets:
        try:
            summary[ds] = run_dataset_text(
                dataset=ds,
                data_root=data_root,
                embed_root=embed_root,
                out_root=out_root,
                num_candidates=args.num_candidates,
                batch_size=args.batch_size,
                device=device,
                k_values=k_values,
            )
        except Exception as e:
            print(f"[{ds}] FAILED: {e}")

    if len(summary) == 3:
        sp = out_root / "summary_text_retrieve.json"
        with sp.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {sp}")
        print("\n=== test 汇总 (text-only) ===")
        ks = k_values
        for ds, m in summary.items():
            te = m.get("test", {})
            parts = [f"H@{k}={100 * te.get(f'hits@{k}', 0):.2f}" for k in ks]
            print(f"{ds}: {' '.join(parts)}  MRR={100 * te.get('mrr', 0):.2f}  n={te.get('n', 0)}")


if __name__ == "__main__":
    main()
