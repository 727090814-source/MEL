# -*- coding: utf-8 -*-
"""
三数据集 CLIP 融合对比学习：仅在训练集上训练，仅在 test 上报告检索指标。

- epochs 默认 20
- eval_splits=test（不计算 train/val 检索，节省时间）
- 输出：各数据集 results/clip_fusion_ep20/<Dataset>/ 与总表 summary_test_ep20.json

依赖 embedding_clip 与 results/clip_retrieve/*/candidate-100.json

用法:
  python scripts/train_clip_fusion_all_three.py
  python scripts/train_clip_fusion_all_three.py --epochs 20 --out_dir results/clip_fusion_ep20
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

# (dataset, candidate_json, lr, weight_decay) — 三数据集统一 lr=1e-3, wd=1e-4
CONFIG = [
    ("WikiMEL", "results/clip_retrieve/WikiMEL/candidate-100.json", 1e-3, 1e-4),
    ("WikiDiverse", "results/clip_retrieve/WikiDiverse/candidate-100.json", 1e-3, 1e-4),
    ("RichpediaMEL", "results/clip_retrieve/RichpediaMEL/candidate-100.json", 1e-3, 1e-4),
]


def main():
    p = argparse.ArgumentParser(description="Fusion training: WikiMEL + WikiDiverse + RichpediaMEL, test-only eval")
    p.add_argument("--data_root", type=str, default="data/raw")
    p.add_argument("--embed_root", type=str, default="embedding_clip")
    p.add_argument("--out_dir", type=str, default="results/clip_fusion_ep20")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--max_negs", type=int, default=100)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--retrieval_batch", type=int, default=256)
    p.add_argument("--k_values", type=str, default="1,5,10,30,100")
    args = p.parse_args()

    summary: dict = {}

    for dataset, cand_rel, lr, wd in CONFIG:
        print("\n" + "=" * 70)
        print(f"  {dataset}  |  lr={lr}  wd={wd}  epochs={args.epochs}")
        print(f"  candidate: {cand_rel}")
        print("=" * 70 + "\n")

        metrics = run_fusion_training(
            dataset=dataset,
            data_root=args.data_root,
            embed_root=args.embed_root,
            candidate_json=cand_rel,
            out_dir=args.out_dir,
            proj_dim=args.proj_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=lr,
            weight_decay=wd,
            max_negs=args.max_negs,
            device_str=args.device,
            retrieval_batch=args.retrieval_batch,
            k_values_str=args.k_values,
            metrics_filename="metrics_test_only.json",
            eval_splits="test",
        )
        summary[dataset] = metrics

    out = Path(args.out_dir) / "summary_test_ep20.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n已保存合并结果: {out}")

    print("\n" + "=" * 70)
    print("  三数据集 test 集结果 (训练集训练 / test 评估)")
    print("=" * 70)
    ks = [int(x) for x in args.k_values.split(",") if x.strip()]
    for ds, m in summary.items():
        te = m.get("test", {})
        parts = [f"H@{k}={100 * te.get(f'hits@{k}', 0):.2f}" for k in ks]
        h = " ".join(parts)
        print(f"{ds}: {h}  MRR={100 * te.get('mrr', 0):.2f}  n={te.get('n', 0)}")


if __name__ == "__main__":
    main()
