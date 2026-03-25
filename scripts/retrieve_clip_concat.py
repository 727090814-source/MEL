# -*- coding: utf-8 -*-
"""
Multimodal retrieval without training: concat CLIP text + image embeddings (512+512=1024),
L2-normalize, cosine retrieval over all KB entities.

With --text_only: only CLIP text vectors (mention vs KB entity text).

With --image_only: only CLIP image vectors (mention image vs KB entity image; missing
images are stored as zeros in run_embedding_clip_mel — those rows retrieve with flat scores).

Outputs (aligned with utils/evaluate.generate_candidate_preds structure):
  {out_root}/{dataset}/candidate-{K}.json
    { "train"|"val"|"test": { "answer", "mention_key", "candidate", "rank" } }
  {out_root}/{dataset}/metrics.json
    hits@k, MRR per split

Requires embedding_clip/{dataset}/ from scripts/run_embedding_clip_mel.py:
  entity_qid2idx.json, mention_key2idx.json always;
  entity_text.pt + mention_text.pt for concat or --text_only;
  entity_img.pt + mention_img.pt for concat or --image_only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Same split naming as run_embedding_clip_mel.py (keep in sync)
_MENTION_SPLIT_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("train.json", "dev.json", "test.json"),
    ("wiki_diverse_train.json", "wiki_diverse_dev.json", "wiki_diverse_test.json"),
    ("RichpediaMEL_train.json", "RichpediaMEL_dev.json", "RichpediaMEL_test.json"),
)
_SPLIT_NAMES = ("train", "dev", "test")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_splits_ordered(ds_root: Path) -> List[Tuple[str, List[dict]]]:
    """Return [(train, items), (dev, items), (test, items)] if full triple exists."""
    for group in _MENTION_SPLIT_GROUPS:
        paths = [ds_root / name for name in group]
        if all(p.exists() for p in paths):
            out: List[Tuple[str, List[dict]]] = []
            for sn, p in zip(_SPLIT_NAMES, paths):
                data = load_json(p)
                items = data if isinstance(data, list) else [data]
                out.append((sn, items))
            return out
    return []


def load_mention_splits(ds_root: Path) -> List[dict]:
    """Same ordering as run_embedding_clip_mel (for row alignment)."""
    items: List[dict] = []
    for group in _MENTION_SPLIT_GROUPS:
        paths = [ds_root / name for name in group]
        if all(p.exists() for p in paths):
            for p in paths:
                data = load_json(p)
                if isinstance(data, list):
                    items.extend(data)
                else:
                    items.append(data)
            return items
    seen: set[str] = set()
    for group in _MENTION_SPLIT_GROUPS:
        for name in group:
            p = ds_root / name
            key = str(p.resolve())
            if p.exists() and key not in seen:
                seen.add(key)
                data = load_json(p)
                if isinstance(data, list):
                    items.extend(data)
                else:
                    items.append(data)
    return items


def build_idx2qid(qid2idx: Dict[str, int]) -> List[str]:
    n = max(qid2idx.values()) + 1 if qid2idx else 0
    idx2qid = [""] * n
    for qid, idx in qid2idx.items():
        idx2qid[int(idx)] = str(qid)
    return idx2qid


def concat_norm(t_text: torch.Tensor, t_img: torch.Tensor) -> torch.Tensor:
    """[N,512] + [N,512] -> [N,1024], L2 normalize rows."""
    x = torch.cat([t_text, t_img], dim=-1)
    return F.normalize(x, p=2, dim=1)


def mention_key(item: dict) -> str:
    return f"{item.get('id')}-{item.get('answer')}"


@torch.no_grad()
def run_retrieval_split(
    mention_emb: torch.Tensor,
    entity_emb: torch.Tensor,
    idx2qid: List[str],
    items: List[dict],
    qid2idx: Dict[str, int],
    device: torch.device,
    batch_size: int,
    k_values: List[int],
    num_candidates: int,
) -> Tuple[Dict[str, List], Dict[str, float]]:
    """
    mention_emb rows must match `items` order (same slice of global mention matrix).
    """
    n_ent = entity_emb.shape[0]
    topk = min(num_candidates, n_ent)
    entity_dev = entity_emb.to(device)

    preds: Dict[str, List] = {
        "answer": [],
        "mention_key": [],
        "candidate": [],
        "rank": [],
    }
    hits = {k: 0 for k in k_values}
    mrr_sum = 0.0
    total = 0
    skipped_oov = 0

    m = len(items)
    if m == 0:
        metrics = {f"hits@{k}": 0.0 for k in k_values}
        metrics["mrr"] = 0.0
        metrics["n"] = 0
        metrics["skipped_oov"] = 0
        return preds, metrics


    for start in tqdm(range(0, m, batch_size), desc="Retrieval", ncols=100, leave=False):
        end = min(start + batch_size, m)
        q = mention_emb[start:end].to(device)
        sim = q @ entity_dev.T
        order = torch.argsort(sim, dim=1, descending=True)
        batch_items = items[start:end]

        for bi, item in enumerate(batch_items):
            ans = str(item.get("answer"))
            gold = qid2idx.get(ans)
            row_order = order[bi]
            top_idx = row_order[:topk].cpu().tolist()
            preds["answer"].append(ans)
            preds["mention_key"].append(mention_key(item))
            preds["candidate"].append([idx2qid[j] for j in top_idx])

            if gold is None:
                skipped_oov += 1
                preds["rank"].append(n_ent + 1)
                continue

            rank_pos = (row_order == gold).nonzero(as_tuple=True)[0]
            if rank_pos.numel() == 0:
                r = n_ent + 1
            else:
                r = int(rank_pos[0].item()) + 1
            mrr_sum += 1.0 / r
            for kk in k_values:
                if r <= kk:
                    hits[kk] += 1
            total += 1
            preds["rank"].append(r)

    metrics = {f"hits@{k}": (hits[k] / total if total else 0.0) for k in k_values}
    metrics["mrr"] = mrr_sum / total if total else 0.0
    metrics["n"] = total
    metrics["skipped_oov"] = skipped_oov
    return preds, metrics


def verify_alignment(
    mention_items: List[dict],
    mention_key2idx: Dict[str, int],
) -> bool:
    for i, item in enumerate(mention_items):
        k = mention_key(item)
        if mention_key2idx.get(k) != i:
            print(f"[warn] mention row mismatch at i={i} key={k} map={mention_key2idx.get(k)}")
            return False
    return True


def run_dataset(
    dataset: str,
    data_root: Path,
    embed_root: Path,
    out_root: Path,
    num_candidates: int,
    batch_size: int,
    device: torch.device,
    k_values: List[int],
    text_only: bool = False,
    image_only: bool = False,
) -> None:
    ds_root = data_root / dataset
    emb_dir = embed_root / dataset
    out_dir = out_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = [emb_dir / "entity_qid2idx.json", emb_dir / "mention_key2idx.json"]
    if text_only:
        req = maps + [emb_dir / "entity_text.pt", emb_dir / "mention_text.pt"]
    elif image_only:
        req = maps + [emb_dir / "entity_img.pt", emb_dir / "mention_img.pt"]
    else:
        req = maps + [
            emb_dir / "entity_text.pt",
            emb_dir / "mention_text.pt",
            emb_dir / "entity_img.pt",
            emb_dir / "mention_img.pt",
        ]
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError(f"[{dataset}] missing:\n" + "\n".join(missing))

    qid2idx: Dict[str, int] = {str(k): int(v) for k, v in load_json(emb_dir / "entity_qid2idx.json").items()}
    mention_key2idx: Dict[str, int] = {
        str(k): int(v) for k, v in load_json(emb_dir / "mention_key2idx.json").items()
    }
    idx2qid = build_idx2qid(qid2idx)

    if text_only:
        et = torch.load(emb_dir / "entity_text.pt", map_location="cpu")
        mt = torch.load(emb_dir / "mention_text.pt", map_location="cpu")
        # run_embedding_clip_mel already L2-normalizes encoded text; eps avoids edge cases
        entity_emb = F.normalize(et.float(), p=2, dim=1, eps=1e-12)
        mention_emb = F.normalize(mt.float(), p=2, dim=1, eps=1e-12)
    elif image_only:
        ei = torch.load(emb_dir / "entity_img.pt", map_location="cpu")
        mi = torch.load(emb_dir / "mention_img.pt", map_location="cpu")
        # zeros = missing image; normalize with eps keeps zeros stable (no NaN)
        entity_emb = F.normalize(ei.float(), p=2, dim=1, eps=1e-12)
        mention_emb = F.normalize(mi.float(), p=2, dim=1, eps=1e-12)
    else:
        et = torch.load(emb_dir / "entity_text.pt", map_location="cpu")
        mt = torch.load(emb_dir / "mention_text.pt", map_location="cpu")
        ei = torch.load(emb_dir / "entity_img.pt", map_location="cpu")
        mi = torch.load(emb_dir / "mention_img.pt", map_location="cpu")
        entity_emb = concat_norm(et.float(), ei.float())
        mention_emb = concat_norm(mt.float(), mi.float())

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

    # Map dev -> val to mirror module/retrieve.py candidate dict keys
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

    if text_only:
        mode = "text-only"
    elif image_only:
        mode = "image-only"
    else:
        mode = "concat"
    for sk, met in metrics_all.items():
        h = " ".join([f"H@{k}={100 * met.get(f'hits@{k}', 0):.2f}" for k in k_values])
        print(f"[{dataset}] [{mode}] {sk} ({met.get('split')}): {h}  MRR={100 * met['mrr']:.2f}  n={met['n']}")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP retrieval: concat, --text_only, or --image_only for MEL datasets"
    )
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--embed_root", type=str, default="embedding_clip")
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="default: clip_retrieve / text_retrieve / image_retrieve by mode",
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Retrieve with CLIP text embeddings only (mention vs KB entity text)",
    )
    parser.add_argument(
        "--image_only",
        action="store_true",
        help="Retrieve with CLIP image embeddings only (mention vs KB entity image)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="WikiMEL | WikiDiverse | RichpediaMEL | all",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=100,
        help="Top-K entity QIDs written to candidate-*.json (retrieval rank / H@k use full KB sort)",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", help="auto: prefer CUDA | cpu | cuda")
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,5,10,16,100",
        help="comma-separated k for Hits@k (default includes 16, 100)",
    )
    args = parser.parse_args()

    if args.text_only and args.image_only:
        parser.error("use at most one of --text_only and --image_only")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    data_root = Path(args.data_root)
    embed_root = Path(args.embed_root)
    if args.out_root is not None:
        out_root = Path(args.out_root)
    elif args.text_only:
        out_root = Path("results/text_retrieve")
    elif args.image_only:
        out_root = Path("results/image_retrieve")
    else:
        out_root = Path("results/clip_retrieve")

    datasets = ["WikiMEL", "WikiDiverse", "RichpediaMEL"] if args.dataset == "all" else [args.dataset]

    if args.text_only:
        mode = "text-only"
    elif args.image_only:
        mode = "image-only"
    else:
        mode = "concat(text+image)"
    print(f"mode={mode}  device={device}  out_root={out_root}  num_candidates={args.num_candidates}  k_values={k_values}")
    if args.device == "auto" and not torch.cuda.is_available():
        print("[note] CUDA not available; using CPU. Install CUDA-enabled PyTorch for GPU.")

    for ds in datasets:
        try:
            run_dataset(
                dataset=ds,
                data_root=data_root,
                embed_root=embed_root,
                out_root=out_root,
                num_candidates=args.num_candidates,
                batch_size=args.batch_size,
                device=device,
                k_values=k_values,
                text_only=args.text_only,
                image_only=args.image_only,
            )
        except Exception as e:
            print(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
