# -*- coding: utf-8 -*-
"""
从 RichpediaMEL 测试集中抽取 100 条做检索检测（Hits@k / MRR）。

依赖：
  - embedding_clip/RichpediaMEL/（与 retrieve_clip_concat 一致）
  - indexed/RichpediaMEL/ 或 data/raw/RichpediaMEL/（train/dev/test + qid2id + kb_entity）

用法：
  python scripts/richpedia_sample100_eval.py
  python scripts/richpedia_sample100_eval.py --seed 0 --write_candidate_subset
  # 仅写出 candidate 子集（无需 torch）：
  python scripts/richpedia_sample100_eval.py --candidate_only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_data_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    repo = Path(__file__).resolve().parents[1]
    for rel in ("indexed", "data/raw"):
        p = repo / rel
        if (p / "RichpediaMEL" / "RichpediaMEL_test.json").exists():
            return p
    return repo / "data/raw"


def subsample_candidate_test(
    cand_path: Path,
    out_path: Path,
    n: int,
    seed: int,
) -> None:
    bundle = _load_json(cand_path)
    te = bundle["test"]
    m = len(te["answer"])
    k = min(n, m)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(m), k=k))
    for key in ("answer", "mention_key", "candidate", "rank"):
        te[key] = [te[key][i] for i in idx]
    bundle["test"] = te
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"已写入子集 candidate（仅 test 截断为 {k} 条）: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="RichpediaMEL: sample 100 test mentions and run retrieval metrics")
    parser.add_argument("--dataset", type=str, default="RichpediaMEL")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="父目录，其下应有 <dataset>/（默认自动在 indexed 与 data/raw 间探测）",
    )
    parser.add_argument("--embed_root", type=str, default="embedding_clip")
    parser.add_argument("--n", type=int, default=100, help="测试抽样条数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k_values", type=str, default="1,5,10,16,100")
    parser.add_argument("--num_candidates", type=int, default=100, help="候选列数（与检索一致）")
    parser.add_argument(
        "--write_candidate_subset",
        action="store_true",
        help="是否从完整 candidate-100.json 写出仅 test 为 100 条的子集文件",
    )
    parser.add_argument(
        "--candidate_only",
        action="store_true",
        help="只生成 candidate 子集文件，不跑 embedding 评估（不需要安装 torch）",
    )
    parser.add_argument(
        "--candidate_src",
        type=str,
        default="results/clip_retrieve/RichpediaMEL/candidate-100.json",
    )
    parser.add_argument(
        "--candidate_out",
        type=str,
        default="results/clip_retrieve/RichpediaMEL/candidate-100_test100.json",
    )
    args = parser.parse_args()

    if args.candidate_only:
        src = Path(args.candidate_src)
        if not src.exists():
            raise FileNotFoundError(f"缺少文件: {src}")
        subsample_candidate_test(src, Path(args.candidate_out), args.n, args.seed)
        return

    import torch
    from retrieve_clip_concat import load_splits_ordered, run_retrieval_split
    from train_clip_fusion_contrastive import load_embeddings_cpu

    root = _resolve_data_root(args.data_root)
    ds_root = root / args.dataset
    emb_dir = Path(args.embed_root) / args.dataset
    if not ds_root.is_dir():
        raise FileNotFoundError(f"缺少数据目录: {ds_root}")
    if not emb_dir.is_dir():
        raise FileNotFoundError(f"缺少 embedding 目录: {emb_dir}")

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    mention_emb, entity_emb, qid2idx, idx2qid = load_embeddings_cpu(emb_dir)
    splits = load_splits_ordered(ds_root)
    if not splits:
        raise RuntimeError(f"未找到 train/dev/test: {ds_root}")
    train_items = splits[0][1]
    dev_items = splits[1][1]
    test_items = splits[2][1]
    n_tr, n_dv, n_te = len(train_items), len(dev_items), len(test_items)
    off = n_tr + n_dv

    if n_te == 0:
        raise RuntimeError("test 为空")
    k = min(args.n, n_te)
    rng = random.Random(args.seed)
    pick = sorted(rng.sample(range(n_te), k=k))

    sub_items = [test_items[i] for i in pick]
    m_test = mention_emb[off + torch.tensor(pick, dtype=torch.long)]

    _, metrics = run_retrieval_split(
        mention_emb=m_test,
        entity_emb=entity_emb,
        idx2qid=idx2qid,
        items=sub_items,
        qid2idx=qid2idx,
        device=device,
        batch_size=args.batch_size,
        k_values=k_values,
        num_candidates=args.num_candidates,
    )

    print("\n========== RichpediaMEL 测试子集检索（抽样） ==========")
    print(f"  seed={args.seed}  n={k}  /  test_total={n_te}")
    for kk in k_values:
        print(f"  hits@{kk} = {100 * metrics.get(f'hits@{kk}', 0):.2f}%")
    print(f"  MRR = {100 * metrics['mrr']:.2f}%")
    print(f"  n_eval = {metrics['n']}  skipped_oov = {metrics.get('skipped_oov', 0)}")

    out_metrics = Path("results/clip_retrieve") / args.dataset / "metrics_test100.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "split": "test",
        "sample_n": k,
        "seed": args.seed,
        "indices_in_test": pick,
        "metrics": metrics,
    }
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存: {out_metrics}")

    if args.write_candidate_subset:
        src = Path(args.candidate_src)
        if not src.exists():
            print(f"[warn] 未找到 {src}，跳过 candidate 子集写出", file=sys.stderr)
        else:
            subsample_candidate_test(src, Path(args.candidate_out), args.n, args.seed)


if __name__ == "__main__":
    main()
