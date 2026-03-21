# -*- coding: utf-8 -*-
"""
从数据集中随机抽样 N 条 mention，并附上 KB 中对应的 gold 实体记录。

输出：results/samples/samples.json（默认）

用法：
  python scripts/export_mention_gold_samples.py
  python scripts/export_mention_gold_samples.py --n 20 --seed 42 --dataset RichpediaMEL
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_richpedia_flat(data_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, dict]]:
    root = data_root / "RichpediaMEL"
    kb = load_json(root / "kb_entity.json")
    if not isinstance(kb, list):
        kb = [kb]
    by_qid: Dict[str, dict] = {}
    for e in kb:
        q = str(e.get("qid") or "").strip()
        if q:
            by_qid[q] = e

    rows: List[Dict[str, Any]] = []
    for split, name in (
        ("train", "RichpediaMEL_train.json"),
        ("dev", "RichpediaMEL_dev.json"),
        ("test", "RichpediaMEL_test.json"),
    ):
        path = root / name
        if not path.exists():
            continue
        for item in load_json(path):
            rows.append({"split": split, **item})
    return rows, by_qid


def main():
    p = argparse.ArgumentParser(description="Sample mentions with gold entities to results/samples/")
    p.add_argument("--dataset", type=str, default="RichpediaMEL", choices=("RichpediaMEL",))
    p.add_argument("--data_root", type=str, default="indexed", help="含 RichpediaMEL/ 的父目录")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_dir",
        type=str,
        default="results/samples",
        help="输出目录（将写入 samples.json）",
    )
    p.add_argument(
        "--filename",
        type=str,
        default="samples.json",
        help="输出文件名",
    )
    args = p.parse_args()

    data_root = Path(args.data_root)
    if args.dataset == "RichpediaMEL":
        all_rows, by_qid = load_richpedia_flat(data_root)
    else:
        raise SystemExit("仅实现 RichpediaMEL")

    if len(all_rows) < args.n:
        raise SystemExit(f"总条数 {len(all_rows)} < 抽样数 {args.n}")

    rng = random.Random(args.seed)
    picked = rng.sample(all_rows, args.n)

    samples: List[Dict[str, Any]] = []
    for it in picked:
        ans = str(it.get("answer") or "").strip()
        gold = by_qid.get(ans)
        samples.append(
            {
                "split": it.get("split"),
                "mention_id": it.get("id"),
                "mention_surface": it.get("mentions"),
                "sentence": it.get("sentence"),
                "imgPath": it.get("imgPath"),
                "gold_qid": ans,
                "field_entities": it.get("entities"),
                "gold_entity": gold if gold is not None else None,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.filename
    payload = {
        "dataset": args.dataset,
        "data_root": str((data_root / "RichpediaMEL").resolve()),
        "n": args.n,
        "seed": args.seed,
        "samples": samples,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已写入 {out_path.resolve()} （{args.n} 条）")


if __name__ == "__main__":
    main()
