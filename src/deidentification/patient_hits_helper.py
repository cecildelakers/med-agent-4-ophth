# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple

from med_deid_common import compact_text, normalize_spaces

# 来自 truncate_ids.py 的核心规则（并做了字段兼容扩展）
DROP_LABELS = {"电话预约", "医院地址"}
TRUNCATE_RULES = {
    "病案号": 6,
    "联系方式": 11,
    "联系电话": 11,
    "联系人电话": 11,
    "门诊病案号": 12,
    "门诊号": 14,
}


def _truncate_value(label: str, raw_value: str) -> str:
    value = normalize_spaces(raw_value)
    limit = TRUNCATE_RULES.get(label)
    if not value or limit is None:
        return value
    return value[:limit].strip()


def build_hit_dedupe_key(row: Dict[str, str]) -> Tuple[str, ...]:
    # 以“患者 + 文件 + 标签 + 规范化值 + 动作”为主键，去掉重复页眉/页脚造成的重复命中。
    return (
        normalize_spaces(row.get("patient_name", "")),
        normalize_spaces(row.get("patient_id_candidate", "")),
        normalize_spaces(row.get("glaucoma_type", "")),
        normalize_spaces(row.get("source_file", "")),
        normalize_spaces(row.get("source_doc_type", "")),
        normalize_spaces(row.get("field_label", "")),
        normalize_spaces(row.get("normalized_value", "")),
        normalize_spaces(row.get("action_suggestion", "")),
    )


def clean_hit_row(row: Dict[str, str]) -> Tuple[Dict[str, str], str]:
    out = dict(row)
    label = normalize_spaces(out.get("field_label", ""))
    if not label:
        return out, "drop_empty_label"
    if label in DROP_LABELS:
        return out, "drop_label"

    raw_value = _truncate_value(label, out.get("raw_value", ""))
    out["field_label"] = label
    out["raw_value"] = raw_value
    out["normalized_value"] = compact_text(raw_value) if raw_value else ""
    out["context"] = normalize_spaces(out.get("context", ""))
    out["patient_name"] = normalize_spaces(out.get("patient_name", ""))
    out["patient_id_candidate"] = normalize_spaces(out.get("patient_id_candidate", ""))
    out["glaucoma_type"] = normalize_spaces(out.get("glaucoma_type", ""))
    out["source_file"] = normalize_spaces(out.get("source_file", ""))
    out["source_doc_type"] = normalize_spaces(out.get("source_doc_type", ""))
    out["action_suggestion"] = normalize_spaces(out.get("action_suggestion", ""))

    # 对 delete_field_value 的空值没有意义，直接丢弃。
    if out["action_suggestion"] == "delete_field_value" and not out["raw_value"]:
        return out, "drop_empty_delete_value"
    return out, "keep"


def clean_patient_sensitive_hits(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    stats = {
        "input_rows": len(rows),
        "kept_rows": 0,
        "dropped_label_rows": 0,
        "dropped_empty_rows": 0,
        "deduped_rows": 0,
        "truncated_rows": 0,
    }
    cleaned: List[Dict[str, str]] = []
    seen = set()

    for row in rows:
        original_raw = normalize_spaces(row.get("raw_value", ""))
        out, status = clean_hit_row(row)
        if status == "drop_label":
            stats["dropped_label_rows"] += 1
            continue
        if status in {"drop_empty_label", "drop_empty_delete_value"}:
            stats["dropped_empty_rows"] += 1
            continue
        if out.get("raw_value", "") != original_raw:
            stats["truncated_rows"] += 1

        key = build_hit_dedupe_key(out)
        if key in seen:
            stats["deduped_rows"] += 1
            continue
        seen.add(key)
        cleaned.append(out)

    stats["kept_rows"] = len(cleaned)
    return cleaned, stats


def dedupe_hits_globally(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    seen = set()
    for row in rows:
        key = build_hit_dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out

