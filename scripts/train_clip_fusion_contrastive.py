# -*- coding: utf-8 -*-
"""
CLIP-style contrastive fusion on frozen concat CLIP embeddings (1024-d).

WikiMEL: InfoNCE with negatives = entity rows for QIDs in candidate file that are
NOT the gold answer (non-matching among top-K retrieved candidates).

After training: project all mentions/entities, full-KB retrieval -> H@k (default includes 30, 100) and MRR.

Example:
  python scripts/train_clip_fusion_contrastive.py \\
    --dataset WikiMEL \\
    --candidate_json results/clip_retrieve/WikiMEL/candidate-100.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from retrieve_clip_concat import (  # noqa: E402
    build_idx2qid,
    concat_norm,
    load_json,
    load_splits_ordered,
    load_mention_splits,
    run_retrieval_split,
    verify_alignment,
)


class CLIPFusion(nn.Module):
    def __init__(self, in_dim: int = 1024, proj_dim: int = 512):
        super().__init__()
        self.m_proj = nn.Linear(in_dim, proj_dim)
        self.e_proj = nn.Linear(in_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_mention(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.m_proj(x), p=2, dim=-1)

    def encode_entity(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.e_proj(x), p=2, dim=-1)


def load_embeddings_cpu(emb_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], List[str]]:
    qid2idx: Dict[str, int] = {str(k): int(v) for k, v in load_json(emb_dir / "entity_qid2idx.json").items()}
    idx2qid = build_idx2qid(qid2idx)
    et = torch.load(emb_dir / "entity_text.pt", map_location="cpu")
    ei = torch.load(emb_dir / "entity_img.pt", map_location="cpu")
    mt = torch.load(emb_dir / "mention_text.pt", map_location="cpu")
    mi = torch.load(emb_dir / "mention_img.pt", map_location="cpu")
    entity_emb = concat_norm(et.float(), ei.float())
    mention_emb = concat_norm(mt.float(), mi.float())
    return mention_emb, entity_emb, qid2idx, idx2qid


def build_train_neg_indices(
    train_candidates: List[List[str]],
    train_answers: List[str],
    qid2idx: Dict[str, int],
    max_negs: int,
) -> List[List[int]]:
    out: List[List[int]] = []
    for cands, ans in zip(train_candidates, train_answers):
        gold = str(ans)
        ne: List[int] = []
        for q in cands:
            qs = str(q)
            if qs == gold:
                continue
            if qs in qid2idx:
                ne.append(qid2idx[qs])
            if len(ne) >= max_negs:
                break
        out.append(ne)
    return out


def infonce_step(
    model: CLIPFusion,
    m: torch.Tensor,
    pos: torch.Tensor,
    neg_block: torch.Tensor,
    neg_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    m, pos: [B, D]
    neg_block: [B, K, D] entity vectors (padded)
    neg_mask: [B, K] 1.0 = valid negative, 0 = pad
    """
    scale = model.logit_scale.exp().clamp(max=100.0)
    m_h = model.encode_mention(m)
    pos_h = model.encode_entity(pos)
    neg_h = model.encode_entity(neg_block.view(-1, neg_block.shape[-1])).view(
        neg_block.shape[0], neg_block.shape[1], -1
    )

    s_pos = (m_h * pos_h).sum(-1, keepdim=True)
    s_neg = (m_h.unsqueeze(1) * neg_h).sum(-1)  # [B, K]
    logits = scale * torch.cat([s_pos, s_neg], dim=1)

    K = neg_mask.shape[1]
    full_mask = torch.cat(
        [torch.ones(m.shape[0], 1, device=device, dtype=logits.dtype), neg_mask], dim=1
    )
    logits = logits.masked_fill(full_mask < 0.5, float("-inf"))
    target = torch.zeros(m.shape[0], dtype=torch.long, device=device)
    return F.cross_entropy(logits, target)


@torch.no_grad()
def project_all(
    model: CLIPFusion,
    mention_emb: torch.Tensor,
    entity_emb: torch.Tensor,
    device: torch.device,
    chunk: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    me = mention_emb.to(device)
    ee = entity_emb.to(device)
    out_m: List[torch.Tensor] = []
    out_e: List[torch.Tensor] = []
    for s in range(0, me.shape[0], chunk):
        out_m.append(model.encode_mention(me[s : s + chunk]).cpu())
    for s in range(0, ee.shape[0], chunk):
        out_e.append(model.encode_entity(ee[s : s + chunk]).cpu())
    return torch.cat(out_m, dim=0), torch.cat(out_e, dim=0)


def run_fusion_training(
    dataset: str,
    data_root: str = "data/raw",
    embed_root: str = "embedding_clip",
    candidate_json: str = "results/clip_retrieve/WikiMEL/candidate-100.json",
    out_dir: str = "results/clip_fusion",
    proj_dim: int = 512,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_negs: int = 100,
    device_str: str = "auto",
    retrieval_batch: int = 256,
    k_values_str: str = "1,5,10,30,100",
    metrics_filename: str = "metrics_after_fusion.json",
    eval_splits: str = "all",
) -> Dict[str, dict]:
    """
    Train CLIP fusion + full-KB retrieval metrics. Returns metrics_all dict.

    eval_splits: "all" -> train/val/test; "test" -> only test split (报告测试集指标).
    """
    device = torch.device(
        device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    k_values = [int(x.strip()) for x in k_values_str.split(",") if x.strip()]

    ds = dataset
    ds_root = Path(data_root) / ds
    emb_dir = Path(embed_root) / ds
    cand_path = Path(candidate_json)
    out_root = Path(out_dir) / ds
    out_root.mkdir(parents=True, exist_ok=True)

    if not cand_path.exists():
        raise FileNotFoundError(
            f"Missing candidate json: {cand_path}\n"
            f"Run: python scripts/retrieve_clip_concat.py --dataset {ds} --num_candidates 100"
        )

    mention_emb, entity_emb, qid2idx, idx2qid = load_embeddings_cpu(emb_dir)
    mention_items = load_mention_splits(ds_root)
    if len(mention_items) != mention_emb.shape[0]:
        raise ValueError("mention / embedding length mismatch")
    if not verify_alignment(mention_items, {str(k): int(v) for k, v in load_json(emb_dir / "mention_key2idx.json").items()}):
        print("[warn] mention_key alignment check failed")

    splits = load_splits_ordered(ds_root)
    if not splits:
        raise RuntimeError("No train/dev/test splits found")
    train_items = splits[0][1]
    dev_items = splits[1][1]
    test_items = splits[2][1]

    cand_bundle = load_json(cand_path)
    train_cands = cand_bundle["train"]["candidate"]
    train_answers = cand_bundle["train"]["answer"]
    if len(train_cands) != len(train_items) or len(train_answers) != len(train_items):
        raise ValueError(
            f"train candidate json length {len(train_cands)} != train.json length {len(train_items)}"
        )

    neg_lists = build_train_neg_indices(train_cands, train_answers, qid2idx, max_negs)
    n_train = len(train_items)
    train_off = 0
    train_m_indices = np.arange(train_off, train_off + n_train, dtype=np.int64)
    train_gold_idx = np.array(
        [qid2idx[str(it.get("answer"))] for it in train_items],
        dtype=np.int64,
    )

    # Pad negatives to fixed K for batching
    K = max((len(x) for x in neg_lists), default=0)
    if K == 0:
        raise RuntimeError("No negatives found — check candidate file and gold answers")
    K = min(K, max_negs)
    neg_pad = torch.zeros(n_train, K, entity_emb.shape[1], dtype=entity_emb.dtype)
    neg_mask = torch.zeros(n_train, K, dtype=torch.float32)
    for i, lst in enumerate(neg_lists):
        for j, eidx in enumerate(lst[:K]):
            neg_pad[i, j] = entity_emb[eidx]
            neg_mask[i, j] = 1.0

    model = CLIPFusion(in_dim=mention_emb.shape[1], proj_dim=proj_dim)
    model.to(device)
    entity_dev = entity_emb.to(device)
    mention_dev = mention_emb.to(device)
    neg_pad = neg_pad.to(device)
    neg_mask = neg_mask.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_m_indices_t = torch.from_numpy(train_m_indices).long().to(device)
    train_gold_idx_t = torch.from_numpy(train_gold_idx).long().to(device)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_steps = 0
        pbar = tqdm(range(0, n_train, batch_size), desc=f"Epoch {ep}/{epochs}", ncols=100)
        for s in pbar:
            idx = perm[s : s + batch_size]
            b_m = train_m_indices_t[idx]
            b_g = train_gold_idx_t[idx]
            m = mention_dev[b_m]
            pos = entity_dev[b_g]
            nb = neg_pad[idx]
            nm = neg_mask[idx]
            loss = infonce_step(model, m, pos, nb, nm, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {ep} mean loss {total_loss / max(n_steps, 1):.6f}")

    ckpt_path = out_root / "fusion_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "proj_dim": proj_dim,
            "in_dim": mention_emb.shape[1],
        },
        ckpt_path,
    )
    print(f"Saved {ckpt_path}")

    # ----- Retrieval on projected embeddings -----
    print("Projecting all mentions/entities for retrieval...")
    m_proj, e_proj = project_all(model, mention_emb, entity_emb, device, chunk=4096)

    split_key = {"train": "train", "dev": "val", "test": "test"}
    split_items_full = [("train", train_items), ("dev", dev_items), ("test", test_items)]
    n_tr, n_dv, n_te = len(train_items), len(dev_items), len(test_items)
    eval_mode = (eval_splits or "all").strip().lower()
    if eval_mode == "test":
        split_items_loop = [("test", test_items)]
        offsets = {"test": n_tr + n_dv}
    else:
        split_items_loop = split_items_full
        offsets = {"train": 0, "dev": n_tr, "test": n_tr + n_dv}

    metrics_all: Dict[str, dict] = {}
    for split_name, items in split_items_loop:
        off = offsets[split_name]
        n = len(items)
        sl = m_proj[off : off + n]
        _, met = run_retrieval_split(
            mention_emb=sl,
            entity_emb=e_proj,
            idx2qid=idx2qid,
            items=items,
            qid2idx=qid2idx,
            device=device,
            batch_size=retrieval_batch,
            k_values=k_values,
            num_candidates=max(100, max(k_values)),
        )
        metrics_all[split_key[split_name]] = {**met, "split": split_name}

    met_path = out_root / metrics_filename
    with met_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)
    print(f"Saved {met_path}")

    for sk, met in metrics_all.items():
        h = " ".join([f"H@{k}={100 * met.get(f'hits@{k}', 0):.2f}" for k in k_values])
        print(f"[{ds}] {sk} ({met.get('split')}): {h}  MRR={100 * met['mrr']:.2f}  n={met['n']}")

    return metrics_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WikiMEL")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--embed_root", type=str, default="embedding_clip")
    parser.add_argument(
        "--candidate_json",
        type=str,
        default="results/clip_retrieve/WikiMEL/candidate-100.json",
        help="Precomputed top-K candidates; train split uses non-gold as negatives",
    )
    parser.add_argument("--out_dir", type=str, default="results/clip_fusion")
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_negs", type=int, default=100, help="Max negatives per sample from candidate pool")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--retrieval_batch", type=int, default=256)
    parser.add_argument("--k_values", type=str, default="1,5,10,30,100")
    parser.add_argument(
        "--eval_splits",
        type=str,
        default="all",
        choices=("all", "test"),
        help="Retrieval eval: all splits or test only",
    )
    args = parser.parse_args()

    run_fusion_training(
        dataset=args.dataset,
        data_root=args.data_root,
        embed_root=args.embed_root,
        candidate_json=args.candidate_json,
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
        eval_splits=args.eval_splits,
    )


if __name__ == "__main__":
    main()
