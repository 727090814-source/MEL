# -*- coding: utf-8 -*-
"""
单独脚本：仅对 WikiDiverse 与 RichpediaMEL 做 CLIP 对比学习融合训练 + 全库检索评估。

依赖：
  - embedding_clip/{WikiDiverse,RichpediaMEL}/
  - results/clip_retrieve/{WikiDiverse,RichpediaMEL}/candidate-100.json
    （若缺失先运行: python scripts/retrieve_clip_concat.py --dataset <NAME> --num_candidates 100）

默认输出目录（与 WikiMEL 主脚本区分）：results/clip_fusion_wd_rp/<dataset>/
  - fusion_model.pt
  - metrics_wd_rp.json

用法：
  python scripts/train_clip_fusion_wd_rp.py
  python scripts/train_clip_fusion_wd_rp.py --epochs 10 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train_clip_fusion_contrastive import run_fusion_training  # noqa: E402


DATASETS = [
    (
        "WikiDiverse",
        "results/clip_retrieve/WikiDiverse/candidate-100.json",
    ),
    (
        "RichpediaMEL",
        "results/clip_retrieve/RichpediaMEL/candidate-100.json",
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Fusion training: WikiDiverse + RichpediaMEL only")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--embed_root", type=str, default="embedding_clip")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/clip_fusion_wd_rp",
        help="Root for per-dataset checkpoints/metrics (default separate from clip_fusion/)",
    )
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4, help="略低于 WikiMEL 默认 1e-3，减轻过拟合")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_negs", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--retrieval_batch", type=int, default=256)
    parser.add_argument("--k_values", type=str, default="1,5,10")
    args = parser.parse_args()

    all_results: dict = {}

    for dataset, cand_rel in DATASETS:
        print("\n" + "=" * 60)
        print(f"  {dataset}  |  candidate: {cand_rel}")
        print("=" * 60 + "\n")

        metrics = run_fusion_training(
            dataset=dataset,
            data_root=args.data_root,
            embed_root=args.embed_root,
            candidate_json=cand_rel,
            out_dir=args.out_dir,
            proj_dim=args.proj_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_negs=args.max_negs,
            device_str=args.device,
            retrieval_batch=args.retrieval_batch,
            k_values_str=args.k_values,
            metrics_filename="metrics_wd_rp.json",
        )
        all_results[dataset] = metrics

    summary_path = Path(args.out_dir) / "summary_wd_rp.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n合并指标已保存: {summary_path}")

    print("\n========== 汇总 (test 集) ==========")
    for ds, m in all_results.items():
        te = m.get("test", {})
        h1 = 100 * te.get("hits@1", 0)
        h5 = 100 * te.get("hits@5", 0)
        h10 = 100 * te.get("hits@10", 0)
        mr = 100 * te.get("mrr", 0)
        print(f"{ds}: H@1={h1:.2f}  H@5={h5:.2f}  H@10={h10:.2f}  MRR={mr:.2f}  n={te.get('n', 0)}")


if __name__ == "__main__":
    main()
