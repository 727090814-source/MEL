import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# 将项目根目录加入 sys.path，确保可以导入 utils 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.encoder import CLIPTextEncoder, CLIPImageEncoder


def load_raw_splits(base_raw_dir: str, dataset: str) -> List[dict]:
    """
    从 data/raw/{dataset} 读取 train/dev/test.json，合并为一个列表。
    """
    root = Path(base_raw_dir) / dataset
    data: List[dict] = []
    for split in ["train", "dev", "test"]:
        path = root / f"{split}.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            part = json.load(f)
        if isinstance(part, list):
            data.extend(part)
        else:
            data.append(part)
    return data


def load_entity_kb(base_raw_dir: str, dataset: str) -> List[dict]:
    """
    从 data/raw/{dataset}/kb_entity.json 读取实体文本信息。
    """
    path = Path(base_raw_dir) / dataset / "kb_entity.json"
    if not path.exists():
        raise FileNotFoundError(f"kb_entity.json not found at {path}")
    with path.open(encoding="utf-8") as f:
        kb = json.load(f)
    if isinstance(kb, list):
        return kb
    return [kb]


def encode_texts(
    texts: List[str],
    out_path: str,
    model_name: str,
    batch_size: int,
) -> torch.Tensor:
    encoder = CLIPTextEncoder(model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    embs = encoder.encode_batch(texts, batch_size=batch_size)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 保存为简单的 torch .pt 文件，避免 h5py 依赖
    torch.save({"embeddings": embs.cpu()}, out_path)
    return embs


def encode_images(
    image_paths: List[Path],
    out_path: str,
    model_name: str,
    batch_size: int,
) -> Tuple[torch.Tensor, List[str]]:
    encoder = CLIPImageEncoder(model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    pil_images = []
    ids: List[str] = []
    for p in tqdm(image_paths, desc="Loading images"):
        try:
            pil_images.append(Image.open(p).convert("RGB"))
            ids.append(p.stem)
        except Exception as e:
            print(f"Skip image {p}: {e}")
            continue
    if not pil_images:
        print("No valid images found, skip image embedding.")
        return torch.empty(0), []
    embs = encoder.encode_batch(pil_images, batch_size=batch_size)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"embeddings": embs.cpu(), "ids": ids}, out_path)
    return embs, ids


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple MEL embedding pipeline (text + image)")
    parser.add_argument("--data_root", type=str, default="data/raw", help="原始 MEL 数据根目录，例如 data/raw")
    parser.add_argument("--dataset", type=str, default="WikiMEL", help="数据集名称：WikiMEL / WikiDiverse / RichpediaMEL")
    parser.add_argument("--image_root", type=str, default="data", help="图片所在的根目录，用于拼接 imgPath")
    parser.add_argument("--base_out", type=str, default="embedding", help="embedding 输出根目录")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP 模型名称")
    parser.add_argument("--batch_size", type=int, default=64, help="编码 batch size")
    args = parser.parse_args()

    # 1) 加载 mention / entity 数据
    print(f"Loading raw splits for {args.dataset} from {args.data_root} ...")
    mentions = load_raw_splits(args.data_root, args.dataset)
    print(f"Total mention items: {len(mentions)}")

    print(f"Loading entity KB for {args.dataset} ...")
    entities = load_entity_kb(args.data_root, args.dataset)
    print(f"Total entities in kb_entity: {len(entities)}")

    # 2) 准备 entity 文本
    entity_texts: List[str] = []
    entity_qids: List[str] = []
    for e in entities:
        qid = e.get("qid") or e.get("QID") or e.get("entity_id")
        if not qid:
            continue
        txt = e.get("text") or e.get("description") or e.get("name") or ""
        entity_qids.append(str(qid))
        entity_texts.append(str(txt))

    print(f"Valid entities with qid: {len(entity_qids)}")

    # 3) 准备 mention 文本（简单版本：mention + sentence）
    mention_keys: List[str] = []
    mention_texts: List[str] = []
    mention_img_full_paths: List[Path] = []

    for item in mentions:
        sent = item.get("sentence", "")
        m = item.get("mentions", "")
        qid = item.get("answer", "")
        key = f"{item.get('id')}-{qid}"
        mention_keys.append(key)
        mention_texts.append(f"{m}: {sent}")

        img_rel = item.get("imgPath", "")
        if img_rel:
            # 直接使用 image_root + 相对路径；如果你的 imgPath 本身是绝对路径，也能兼容
            img_path = Path(args.image_root) / img_rel if not os.path.isabs(img_rel) else Path(img_rel)
            mention_img_full_paths.append(img_path)
        else:
            mention_img_full_paths.append(None)

    print(f"Total mention texts: {len(mention_texts)}")

    # 4) 编码 entity 文本 / mention 文本
    entity_text_path = os.path.join(args.base_out, "entity", f"{args.dataset}_entity_text.pt")
    mention_text_path = os.path.join(args.base_out, "mention", f"{args.dataset}_mention_text.pt")

    print(f"Encoding entity texts -> {entity_text_path}")
    encode_texts(entity_texts, entity_text_path, args.model_name, args.batch_size)

    print(f"Encoding mention texts -> {mention_text_path}")
    encode_texts(mention_texts, mention_text_path, args.model_name, args.batch_size)

    # 5) 保存 mapping（qid -> idx, mention_key -> idx）
    mapping_dir = os.path.join(args.base_out, "mention")
    os.makedirs(mapping_dir, exist_ok=True)
    mapping_path = os.path.join(mapping_dir, f"{args.dataset}_mention_text_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "entity_qid2idx": {qid: i for i, qid in enumerate(entity_qids)},
                "mention_key2idx": {k: i for i, k in enumerate(mention_keys)},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved mapping -> {mapping_path}")

    # 6) 编码 mention 图像（如果路径有效）
    valid_img_paths = [p for p in mention_img_full_paths if p is not None and p.exists()]
    if valid_img_paths:
        mention_img_path = os.path.join(args.base_out, "mention", f"{args.dataset}_mention_img.pt")
        print(f"Encoding {len(valid_img_paths)} mention images -> {mention_img_path}")
        encode_images(valid_img_paths, mention_img_path, args.model_name, args.batch_size)
    else:
        print("No valid mention images found, skip image embedding.")

    print("Embedding generation finished.")


if __name__ == "__main__":
    main()

