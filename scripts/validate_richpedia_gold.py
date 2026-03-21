# -*- coding: utf-8 -*-
"""
RichpediaMEL 数据自检：判断 gold（answer QID）是否在 KB 内、与 kb_entity 名称及
mention/sentence 等字段是否一致。不调用外部 API，仅做数据集内部一致性检查。

用法：
  python scripts/validate_richpedia_gold.py
  python scripts/validate_richpedia_gold.py --data_root indexed --report results/diagnostics/richpedia_gold_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "").strip())


def _norm_cf(s: str) -> str:
    return _norm(s).casefold()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_kb_maps(kb: List[dict]) -> Tuple[Dict[str, dict], Dict[str, str]]:
    by_qid: Dict[str, dict] = {}
    name_by_qid: Dict[str, str] = {}
    for e in kb:
        q = str(e.get("qid") or "").strip()
        if not q:
            continue
        by_qid[q] = e
        name_by_qid[q] = str(e.get("entity_name") or "")
    return by_qid, name_by_qid


def mention_in_sentence(mention: str, sentence: str) -> bool:
    m, s = _norm_cf(mention), _norm_cf(sentence)
    if not m:
        return True
    if m in s:
        return True
    # 宽松：去掉空白与常见标点后再比一次
    m2 = re.sub(r"[\s\u00a0]+", " ", m)
    s2 = re.sub(r"[\s\u00a0]+", " ", s)
    if m2 in s2:
        return True
    return False


def main():
    p = argparse.ArgumentParser(description="Validate RichpediaMEL gold labels (internal consistency)")
    p.add_argument("--data_root", type=str, default="indexed", help="含 RichpediaMEL/ 的父目录")
    p.add_argument(
        "--report",
        type=str,
        default="results/diagnostics/richpedia_gold_report.json",
        help="写出 JSON 报告路径",
    )
    p.add_argument("--max_examples", type=int, default=15, help="每种问题最多打印/记录样例条数")
    args = p.parse_args()

    root = Path(args.data_root) / "RichpediaMEL"
    if not root.is_dir():
        raise FileNotFoundError(root)

    qid2id: Dict[str, int] = {str(k): int(v) for k, v in load_json(root / "qid2id.json").items()}
    kb = load_json(root / "kb_entity.json")
    if not isinstance(kb, list):
        kb = [kb]
    _, name_by_qid = build_kb_maps(kb)

    splits = ("train", "dev", "test")
    files = {s: root / f"RichpediaMEL_{s}.json" for s in splits}

    summary: Dict[str, Any] = {
        "dataset": "RichpediaMEL",
        "data_root": str(root.resolve()),
        "kb_entities": len(kb),
        "qid2id_size": len(qid2id),
        "splits": {},
    }

    all_keys: List[str] = []

    for split in splits:
        items: List[dict] = load_json(files[split])
        n = len(items)
        not_in_kb = 0
        name_mismatch = 0
        mention_not_in_sent = 0
        examples: Dict[str, List[dict]] = {
            "gold_not_in_qid2id": [],
            "entity_name_vs_entities_field": [],
            "mention_not_in_sentence": [],
        }

        seen = Counter()
        for it in items:
            k = f"{it.get('id')}-{it.get('answer')}"
            seen[k] += 1
            all_keys.append(k)

            ans = str(it.get("answer") or "").strip()
            if ans not in qid2id:
                not_in_kb += 1
                if len(examples["gold_not_in_qid2id"]) < args.max_examples:
                    examples["gold_not_in_qid2id"].append(
                        {"id": it.get("id"), "answer": ans, "split": split}
                    )
                continue

            kb_name = _norm(name_by_qid.get(ans, ""))
            ent_field = _norm(str(it.get("entities") or ""))
            if kb_name and ent_field and _norm_cf(kb_name) != _norm_cf(ent_field):
                name_mismatch += 1
                if len(examples["entity_name_vs_entities_field"]) < args.max_examples:
                    examples["entity_name_vs_entities_field"].append(
                        {
                            "id": it.get("id"),
                            "answer": ans,
                            "kb_entity_name": kb_name,
                            "field_entities": ent_field,
                            "split": split,
                        }
                    )

            if not mention_in_sentence(str(it.get("mentions") or ""), str(it.get("sentence") or "")):
                mention_not_in_sent += 1
                if len(examples["mention_not_in_sentence"]) < args.max_examples:
                    examples["mention_not_in_sentence"].append(
                        {
                            "id": it.get("id"),
                            "mentions": it.get("mentions"),
                            "sentence": it.get("sentence"),
                            "split": split,
                        }
                    )

        dup_internal = sum(1 for c in seen.values() if c > 1)

        summary["splits"][split] = {
            "n": n,
            "gold_not_in_qid2id": not_in_kb,
            "kb_name_vs_entities_mismatch": name_mismatch,
            "mention_not_in_sentence": mention_not_in_sent,
            "duplicate_keys_in_split": dup_internal,
            "examples": examples,
        }

    # 跨划分重复 mention_key
    key_counts = Counter(all_keys)
    cross_dups = [k for k, v in key_counts.items() if v > 1]
    summary["cross_split_duplicate_keys"] = len(cross_dups)
    summary["cross_split_duplicate_examples"] = cross_dups[: args.max_examples]

    # 图像路径（相对 ds_root）
    img_missing = 0
    img_checked = 0
    img_examples: List[dict] = []
    for split in splits:
        for it in load_json(files[split]):
            rel = it.get("mention_image_path") or it.get("imgPath") or ""
            if not rel:
                continue
            img_checked += 1
            path = root / str(rel)
            if not path.is_file():
                img_missing += 1
                if len(img_examples) < args.max_examples:
                    img_examples.append(
                        {"split": split, "id": it.get("id"), "path": str(rel), "resolved": str(path)}
                    )
    summary["mention_image_files"] = {
        "rows_with_path": img_checked,
        "missing_file": img_missing,
        "examples": img_examples,
    }

    # 结论（面向「gold 是否正确」）
    verdict: List[str] = []
    if summary["splits"]["train"]["gold_not_in_qid2id"] == 0 and summary["splits"]["dev"]["gold_not_in_qid2id"] == 0:
        if summary["splits"]["test"]["gold_not_in_qid2id"] == 0:
            verdict.append("所有划分的 gold QID 均在 qid2id / KB 映射内（无 OOV 标签）。")
    else:
        verdict.append("存在 gold QID 不在 qid2id 中，需人工核对。")

    verdict.append(
        "kb_entity.entity_name 与样本字段 entities 不一致的条数见各 split 的 kb_name_vs_entities_mismatch；"
        "多为别名/消歧写法差异，不代表 gold QID 错误。"
    )
    verdict.append(
        "mentions 未以子串形式出现在 sentence 中的条数见 mention_not_in_sentence；"
        "可能为多词提及、标点或标注口径导致，需抽样人工看句。"
    )
    summary["verdict_notes"] = verdict

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("======== RichpediaMEL gold 自检 ========")
    for split in splits:
        s = summary["splits"][split]
        print(
            f"[{split}] n={s['n']}  gold不在qid2id={s['gold_not_in_qid2id']}  "
            f"名称字段不一致={s['kb_name_vs_entities_mismatch']}  "
            f"mention不在句中={s['mention_not_in_sentence']}  划分内重复key行数={s['duplicate_keys_in_split']}"
        )
    print(f"跨划分重复 mention_key 种类数: {summary['cross_split_duplicate_keys']}")
    mi = summary["mention_image_files"]
    print(f"mention 图像: 有路径行数={mi['rows_with_path']}  文件缺失={mi['missing_file']}")
    print(f"\n详细报告: {out.resolve()}")
    for line in verdict:
        print(f"  · {line}")


if __name__ == "__main__":
    main()
