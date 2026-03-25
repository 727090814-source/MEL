# -*- coding: utf-8 -*-
"""
Run real CLIP embedding for MEL (text + image).

This script does NOT rely on the old KGMEL h5py pipeline.
It produces torch .pt tensors + json mappings:

embedding_clip/{dataset}/
  entity_text.pt        [num_entities, 512]
  entity_img.pt         [num_entities, 512]   (zero if missing)
  mention_text.pt       [num_mentions, 512]
  mention_img.pt        [num_mentions, 512]    (zero if missing)
  entity_qid2idx.json   (qid -> idx) from qid2id.json
  mention_key2idx.json  ((id-answerQID) -> idx)

Paths are resolved from:
  data_root/{dataset}/kb_entity.json
  data_root/{dataset}/image/* or kb_image/* (uses first directory that exists)
  data_root/{dataset}/{train,dev,test}.json  (WikiMEL)
  or wiki_diverse_{train,dev,test}.json (WikiDiverse)
  or RichpediaMEL_{train,dev,test}.json (RichpediaMEL)
  mention_image_path / imgPath inside each sample (relative to data_root/{dataset})
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPImageProcessor
import torch.nn.functional as F


EMBED_DIM = 512


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# train/dev/test JSON naming varies by dataset release
_MENTION_SPLIT_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("train.json", "dev.json", "test.json"),
    ("wiki_diverse_train.json", "wiki_diverse_dev.json", "wiki_diverse_test.json"),
    ("RichpediaMEL_train.json", "RichpediaMEL_dev.json", "RichpediaMEL_test.json"),
)


def load_mention_splits(ds_root: Path) -> List[dict]:
    """Load mention splits; returns [] if no known split files exist."""
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
    # Partial fallback: load any known filename that exists (avoid duplicates)
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


def encode_texts(
    model: CLIPModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Returns normalized embeddings [N, 512] on CPU.
    """
    if not texts:
        return torch.empty((0, EMBED_DIM), dtype=torch.float32)
    outs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text batches", ncols=100):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=77,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model.get_text_features(**inputs)  # [B, 512]
            emb = F.normalize(emb, p=2, dim=1)
            outs.append(emb.detach().cpu())
    return torch.cat(outs, dim=0)


def encode_images(
    model: CLIPModel,
    processor: CLIPImageProcessor,
    image_paths: List[Path],
    device: str,
    batch_size: int,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Returns normalized embeddings [N, 512] on CPU and image ids (stem) in order.
    """
    image_ids: List[str] = []
    pil_images: List[Image.Image] = []
    keep_indices: List[int] = []
    for idx, p in enumerate(tqdm(image_paths, desc="Loading images", ncols=100)):
        try:
            pil_images.append(Image.open(p).convert("RGB"))
            image_ids.append(p.stem)
            keep_indices.append(idx)
        except Exception:
            # skip unreadable image
            continue

    if not pil_images:
        return torch.empty((0, EMBED_DIM), dtype=torch.float32), []

    outs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(pil_images), batch_size), desc="Encoding image batches", ncols=100):
            batch_imgs = pil_images[i : i + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model.get_image_features(**inputs)  # [B, 512]
            emb = F.normalize(emb, p=2, dim=1)
            outs.append(emb.detach().cpu())
    return torch.cat(outs, dim=0), image_ids


def resolve_entity_image_dir(ds_root: Path) -> Path | None:
    """Prefer ``image/``, then ``kb_image/`` for KB entity image files."""
    for sub in ("image", "kb_image"):
        d = ds_root / sub
        if d.is_dir():
            return d
    return None


def mel_project_root(ds_root: Path) -> Path:
    """data/raw/<Dataset> -> project root (MEL)."""
    return ds_root.parents[2]


def resolve_mention_image_path(ds_root: Path, dataset: str, rel: str) -> Path | None:
    """
    Resolve mention image path: json may give only a basename while file lives under
    mention_image/ or mention_images/ (RichpediaMEL, WikiDiverse), or under parallel
    image/<Dataset>/... in the repo.
    """
    rel = (rel or "").strip()
    if not rel:
        return None
    p = Path(rel)
    mel = mel_project_root(ds_root)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        base = p.name
        candidates.extend(
            [
                ds_root / p,
                ds_root / "mention_image" / base,
                ds_root / "mention_images" / base,
                ds_root / "mention_image" / "mention_image" / base,
                ds_root / "mention_images" / "mention_images" / base,
                mel / "image" / dataset / "mention_image" / base,
                mel / "image" / dataset / "mention_images" / base,
                mel / "image" / dataset / "mention_image" / "mention_image" / base,
                mel / "image" / dataset / "mention_images" / "mention_images" / base,
            ]
        )
    for c in candidates:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def extract_qid_from_img_stem(stem: str) -> str | None:
    # expected: Q123_0 or Q123_1
    m = re.match(r"^(Q\d+)_", stem)
    if not m:
        return None
    return m.group(1)


def main():
    parser = argparse.ArgumentParser(description="Real CLIP embedding for MEL (text + image)")
    parser.add_argument("--data_root", type=str, default="data/raw", help="data/raw")
    parser.add_argument("--dataset", type=str, required=True, help="WikiMEL / WikiDiverse / RichpediaMEL")
    parser.add_argument("--model_name", type=str, required=True, help="Local CLIP path, e.g. D:\\models\\clip-vit-base-patch32")
    parser.add_argument("--out_root", type=str, default="embedding_clip", help="Output root")
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--image_batch_size", type=int, default=64)
    parser.add_argument("--no_entity_img", action="store_true", help="Skip entity image embedding")
    parser.add_argument("--no_mention_img", action="store_true", help="Skip mention image embedding")
    args = parser.parse_args()

    ds_root = Path(args.data_root) / args.dataset
    out_root = Path(args.out_root) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.dataset}] device={device}")

    # 1) load qid2id (entity index mapping)
    qid2id_path = ds_root / "qid2id.json"
    qid2id: Dict[str, int] = load_json(qid2id_path)
    num_entities = len(qid2id)
    (out_root / "entity_qid2idx.json").write_text(
        json.dumps({str(k): int(v) for k, v in qid2id.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[{args.dataset}] num_entities={num_entities}")

    # 2) load kb_entity for entity text
    kb_entity = load_json(ds_root / "kb_entity.json")
    if not isinstance(kb_entity, list):
        kb_entity = [kb_entity]

    entity_text = torch.zeros((num_entities, EMBED_DIM), dtype=torch.float32)

    # build texts in the qid2id index order
    # but we don't know which ids exist in kb_entity, so we encode only valid qids and scatter
    valid_qids: List[str] = []
    valid_texts: List[str] = []
    for e in kb_entity:
        qid = str(e.get("qid") or "").strip()
        if not qid or qid not in qid2id:
            continue
        name = str(e.get("entity_name") or "")
        instance = str(e.get("instance") or "")
        attr = str(e.get("attr") or "")
        text = f"{name} {instance} {attr}".strip()
        valid_qids.append(qid)
        valid_texts.append(text)

    print(f"[{args.dataset}] entity_text items={len(valid_qids)}")

    # init CLIP
    print(f"[{args.dataset}] Loading CLIP model from {args.model_name} ...")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    processor = CLIPImageProcessor.from_pretrained(args.model_name)

    # 3) entity text embedding
    print(f"[{args.dataset}] Encoding entity text ...")
    entity_text_emb = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=valid_texts,
        device=device,
        batch_size=args.text_batch_size,
    )
    for qid, emb in zip(valid_qids, entity_text_emb):
        entity_text[qid2id[qid]] = emb
    torch.save(entity_text, out_root / "entity_text.pt")
    print(f"[{args.dataset}] Saved entity_text.pt")

    # 4) load mention splits (WikiMEL / WikiDiverse / RichpediaMEL file names)
    mention_items = load_mention_splits(ds_root)
    print(f"[{args.dataset}] mention items={len(mention_items)}")

    mention_key2idx: Dict[str, int] = {}
    mention_texts: List[str] = []
    mention_keys: List[str] = []
    mention_img_paths: List[Path | None] = []

    for i, item in enumerate(mention_items):
        mid = str(item.get("id"))
        ans = str(item.get("answer"))
        mention_key = f"{mid}-{ans}"
        mention_key2idx[mention_key] = i
        mention_keys.append(mention_key)

        sent = str(item.get("sentence") or "")
        m = str(item.get("mentions") or "")
        mention_texts.append(f"{m}: {sent}".strip())

        rel = item.get("mention_image_path") or item.get("imgPath") or ""
        mention_img_paths.append(resolve_mention_image_path(ds_root, args.dataset, str(rel)))

    (out_root / "mention_key2idx.json").write_text(
        json.dumps(mention_key2idx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 5) mention text embedding
    print(f"[{args.dataset}] Encoding mention text ...")
    mention_text = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=mention_texts,
        device=device,
        batch_size=args.text_batch_size,
    )
    torch.save(mention_text, out_root / "mention_text.pt")
    print(f"[{args.dataset}] Saved mention_text.pt")

    # 6) entity image embedding (optional)
    if not args.no_entity_img:
        print(f"[{args.dataset}] Encoding entity images ...")
        kb_image_dir = resolve_entity_image_dir(ds_root)
        if kb_image_dir is None:
            print(f"[{args.dataset}] Neither image/ nor kb_image/ found under {ds_root}, skip entity images.")
        else:
            print(f"[{args.dataset}] Entity image dir: {kb_image_dir}")
            image_paths = sorted(
                [p for p in kb_image_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            )
            entity_sum = torch.zeros((num_entities, EMBED_DIM), dtype=torch.float32)
            entity_cnt = torch.zeros((num_entities,), dtype=torch.int64)

            # Encode in batches, but encode_images loads all images into RAM for PIL.
            # Since kb_image count is moderate, it's OK.
            img_embs, img_ids = encode_images(
                model=model,
                processor=processor,
                image_paths=image_paths,
                device=device,
                batch_size=args.image_batch_size,
            )
            # img_ids order corresponds to embeddings order in encode_images output
            for stem, emb in zip(img_ids, img_embs):
                qid = extract_qid_from_img_stem(stem)
                if not qid or qid not in qid2id:
                    continue
                idx = qid2id[qid]
                entity_sum[idx] += emb
                entity_cnt[idx] += 1
            # avoid div by zero
            entity_img = torch.zeros((num_entities, EMBED_DIM), dtype=torch.float32)
            nonzero = entity_cnt > 0
            if nonzero.any():
                entity_img[nonzero] = entity_sum[nonzero] / entity_cnt[nonzero].unsqueeze(1).float()
            torch.save(entity_img, out_root / "entity_img.pt")
            print(f"[{args.dataset}] Saved entity_img.pt")
    else:
        print(f"[{args.dataset}] Skip entity_img per --no_entity_img")

    # 7) mention image embedding (optional)
    if not args.no_mention_img:
        print(f"[{args.dataset}] Encoding mention images ...")
        mention_img = torch.zeros((len(mention_items), EMBED_DIM), dtype=torch.float32)

        # encode only existing images, but keep alignment by indices
        valid_paths: List[Path] = []
        valid_indices: List[int] = []
        for i, p in enumerate(mention_img_paths):
            if p is None:
                continue
            valid_paths.append(p)
            valid_indices.append(i)

        if valid_paths:
            img_embs, _ = encode_images(
                model=model,
                processor=processor,
                image_paths=valid_paths,
                device=device,
                batch_size=args.image_batch_size,
            )
            # encode_images keeps order of successfully loaded images.
            # Since we skip unreadable, valid_indices must match successfully loaded count.
            # For safety, if counts mismatch, we only fill the min.
            n = min(len(valid_indices), img_embs.shape[0])
            mention_img[valid_indices[:n]] = img_embs[:n]
        torch.save(mention_img, out_root / "mention_img.pt")
        print(f"[{args.dataset}] Saved mention_img.pt")
    else:
        print(f"[{args.dataset}] Skip mention_img per --no_mention_img")

    print(f"[{args.dataset}] DONE. Output at {out_root}")


if __name__ == "__main__":
    main()

