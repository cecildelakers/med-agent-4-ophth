# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from med_deid_common import (
    ALL_KNOWN_LABELS,
    DOCTOR_TITLE_WORDS,
    DOCTOR_TRIGGER_LABELS,
    HOSPITAL_LABELS,
    PATIENT_NAME_LABELS,
    REMOVE_VALUE_LABELS,
    compact_text,
    discover_patients,
    document_to_blocks,
    extract_docx_images_in_order,
    extract_kv_segments,
    get_ocr_engine,
    infer_doc_type,
    make_context_snippet,
    normalize_hospital_name,
    normalize_person_name,
    normalize_spaces,
    ocr_image_bytes,
    safe_read_csv,
    setup_logger,
    write_csv,
)
from patient_hits_helper import (
    build_hit_dedupe_key,
    clean_patient_sensitive_hits,
    dedupe_hits_globally,
)

DOCTOR_LABEL_PATTERN = re.compile(
    rf"(?P<label>{'|'.join(re.escape(x) for x in sorted(DOCTOR_TRIGGER_LABELS, key=len, reverse=True))})\s*[：:]\s*(?P<name>[\u4e00-\u9fff]{{2,4}})",
    re.IGNORECASE,
)
DOCTOR_TITLE_PATTERN = re.compile(
    rf"(?P<name>[\u4e00-\u9fff]{{2,4}})\s*(?P<title>{'|'.join(re.escape(x) for x in sorted(DOCTOR_TITLE_WORDS, key=len, reverse=True))})",
    re.IGNORECASE,
)
HOSPITAL_GENERIC_PATTERN = re.compile(r"(?P<name>[A-Za-z0-9\u4e00-\u9fff]{2,40}(?:眼科医院|医院|电子病历系统))")

PATIENT_FIELD_ACTIONS = {label: "delete_field_value" for label in REMOVE_VALUE_LABELS}
PATIENT_FIELD_ACTIONS.update({label: "replace_with_patient_id" for label in PATIENT_NAME_LABELS})

PATIENT_REGISTRY_FIELDS = ["patient_name", "patient_id", "glaucoma_type", "patient_dir"]
PATIENT_HIT_FIELDS = [
    "patient_name", "patient_id_candidate", "glaucoma_type", "source_file", "source_doc_type",
    "record_index", "field_label", "raw_value", "normalized_value", "context", "action_suggestion",
]
DOCTOR_CANDIDATE_FIELDS = [
    "candidate_name", "count", "patient_count", "contexts", "trigger_labels",
    "source_doc_types", "example_paths", "confidence", "doctor_id", "confirmed", "notes",
]
HOSPITAL_CANDIDATE_FIELDS = [
    "candidate_name", "count", "patient_count", "contexts", "trigger_labels",
    "source_doc_types", "example_paths", "confidence", "hospital_id", "confirmed", "notes",
]
DOCTOR_TEMPLATE_FIELDS = [
    "candidate_name", "doctor_id", "confirmed", "notes", "count", "patient_count",
    "contexts", "trigger_labels", "source_doc_types", "example_paths", "confidence",
]
HOSPITAL_TEMPLATE_FIELDS = [
    "candidate_name", "hospital_id", "confirmed", "notes", "count", "patient_count",
    "contexts", "trigger_labels", "source_doc_types", "example_paths", "confidence",
]
PROCESSED_FIELDS = ["glaucoma_type", "patient_name", "patient_id", "processed_at"]


def add_candidate(store: Dict[str, dict], candidate: str, context: str, trigger_label: str, confidence: float,
                  doc_type: str, patient_name: str, source_path: str):
    if not candidate:
        return
    if candidate not in store:
        store[candidate] = {
            "candidate_name": candidate,
            "count": 0,
            "contexts": set(),
            "trigger_labels": set(),
            "source_doc_types": set(),
            "patient_names": set(),
            "example_paths": set(),
            "confidences": [],
        }
    row = store[candidate]
    row["count"] += 1
    if context:
        row["contexts"].add(context[:160])
    if trigger_label:
        row["trigger_labels"].add(trigger_label)
    if doc_type:
        row["source_doc_types"].add(doc_type)
    if patient_name:
        row["patient_names"].add(patient_name)
    if source_path:
        row["example_paths"].add(source_path)
    row["confidences"].append(float(confidence))


def scan_doctor_candidates(text: str, doc_type: str, patient_name: str, source_path: str, store: Dict[str, dict]) -> None:
    txt = normalize_spaces(text)
    if not txt:
        return
    for m in DOCTOR_LABEL_PATTERN.finditer(txt):
        name = normalize_person_name(m.group("name"))
        if name:
            add_candidate(
                store,
                name,
                make_context_snippet(txt, m.start(), m.end()),
                m.group("label"),
                0.99,
                doc_type,
                patient_name,
                source_path,
            )
    for m in DOCTOR_TITLE_PATTERN.finditer(txt):
        name = normalize_person_name(m.group("name"))
        if name:
            add_candidate(
                store,
                name,
                make_context_snippet(txt, m.start(), m.end()),
                m.group("title"),
                0.93,
                doc_type,
                patient_name,
                source_path,
            )


def scan_hospital_candidates(text: str, doc_type: str, patient_name: str, source_path: str, store: Dict[str, dict]) -> None:
    txt = normalize_spaces(text)
    if not txt:
        return
    for label, value, start, end in extract_kv_segments(txt, HOSPITAL_LABELS):
        if value:
            name = normalize_hospital_name(value)
            if len(name) >= 3:
                add_candidate(store, name, make_context_snippet(txt, start, end), label, 0.95, doc_type, patient_name, source_path)
    for m in HOSPITAL_GENERIC_PATTERN.finditer(txt):
        cand = normalize_hospital_name(m.group("name"))
        if not cand:
            continue
        bad = {"门诊病历", "门诊", "病历", "医院地址", "电子病历", "病案号", "门诊号"}
        if cand in bad:
            continue
        conf = 0.78
        ctx = make_context_snippet(txt, m.start(), m.end())
        if "公众号" in ctx or "地址" in ctx or "病历系统" in ctx:
            conf = 0.88
        add_candidate(store, cand, ctx, "generic_hospital_pattern", conf, doc_type, patient_name, source_path)


def scan_patient_sensitive_hits(text: str, patient_name: str, patient_id: str, glaucoma_type: str, source_file: str,
                                source_doc_type: str, record_index: int) -> List[dict]:
    txt = normalize_spaces(text)
    hits = []
    if patient_name and patient_name in txt:
        pos = txt.find(patient_name)
        hits.append({
            "patient_name": patient_name,
            "patient_id_candidate": patient_id,
            "glaucoma_type": glaucoma_type,
            "source_file": source_file,
            "source_doc_type": source_doc_type,
            "record_index": record_index,
            "field_label": "patient_name_literal",
            "raw_value": patient_name,
            "normalized_value": patient_name,
            "context": make_context_snippet(txt, pos, pos + len(patient_name)),
            "action_suggestion": "replace_with_patient_id",
        })
    segments = extract_kv_segments(txt, list(PATIENT_FIELD_ACTIONS.keys()) + DOCTOR_TRIGGER_LABELS + HOSPITAL_LABELS + ALL_KNOWN_LABELS)
    for label, value, start, end in segments:
        if label not in PATIENT_FIELD_ACTIONS or not value:
            continue
        hits.append({
            "patient_name": patient_name,
            "patient_id_candidate": patient_id,
            "glaucoma_type": glaucoma_type,
            "source_file": source_file,
            "source_doc_type": source_doc_type,
            "record_index": record_index,
            "field_label": label,
            "raw_value": value,
            "normalized_value": compact_text(value),
            "context": make_context_snippet(txt, start, end),
            "action_suggestion": PATIENT_FIELD_ACTIONS[label],
        })
    return hits


def finalize_candidate_rows(store: Dict[str, dict], id_field_name: str) -> List[dict]:
    rows = []
    for _, item in sorted(store.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
        conf = sum(item["confidences"]) / max(1, len(item["confidences"]))
        rows.append({
            "candidate_name": item["candidate_name"],
            "count": item["count"],
            "patient_count": len(item["patient_names"]),
            "contexts": " || ".join(sorted(item["contexts"])[:5]),
            "trigger_labels": "|".join(sorted(item["trigger_labels"])),
            "source_doc_types": "|".join(sorted(item["source_doc_types"])),
            "example_paths": " | ".join(sorted(item["example_paths"])[:5]),
            "confidence": f"{conf:.3f}",
            id_field_name: "",
            "confirmed": "0",
            "notes": "",
        })
    return rows


def _patient_key(glaucoma_type: str, patient_name: str) -> str:
    return f"{normalize_spaces(glaucoma_type)}::{normalize_spaces(patient_name)}"


def _safe_int(v, default=0) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _safe_float(v, default=0.0) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _split_pipe(s: str) -> List[str]:
    txt = normalize_spaces(s)
    if not txt:
        return []
    return [x for x in txt.split("|") if x]


def _split_contexts(s: str) -> List[str]:
    txt = normalize_spaces(s)
    if not txt:
        return []
    return [x for x in txt.split(" || ") if x]


def _split_examples(s: str) -> List[str]:
    txt = normalize_spaces(s)
    if not txt:
        return []
    return [x for x in txt.split(" | ") if x]


def _serialize_store(store: Dict[str, dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for k, v in store.items():
        out[k] = {
            "candidate_name": v.get("candidate_name", k),
            "count": int(v.get("count", 0)),
            "contexts": sorted(v.get("contexts", set())),
            "trigger_labels": sorted(v.get("trigger_labels", set())),
            "source_doc_types": sorted(v.get("source_doc_types", set())),
            "patient_names": sorted(v.get("patient_names", set())),
            "example_paths": sorted(v.get("example_paths", set())),
            "confidences": [float(x) for x in v.get("confidences", [])],
        }
    return out


def _deserialize_store(obj: dict) -> Dict[str, dict]:
    store: Dict[str, dict] = {}
    if not isinstance(obj, dict):
        return store
    for k, v in obj.items():
        if not isinstance(v, dict):
            continue
        candidate_name = normalize_spaces(v.get("candidate_name", k))
        if not candidate_name:
            continue
        confidences = [float(x) for x in v.get("confidences", [])]
        store[candidate_name] = {
            "candidate_name": candidate_name,
            "count": int(v.get("count", 0)),
            "contexts": set(v.get("contexts", [])),
            "trigger_labels": set(v.get("trigger_labels", [])),
            "source_doc_types": set(v.get("source_doc_types", [])),
            "patient_names": set(v.get("patient_names", [])),
            "example_paths": set(v.get("example_paths", [])),
            "confidences": confidences,
        }
    return store


def _candidate_store_from_rows(rows: List[dict]) -> Dict[str, dict]:
    store: Dict[str, dict] = {}
    for row in rows:
        cand = normalize_spaces(row.get("candidate_name", ""))
        if not cand:
            continue
        count = max(0, _safe_int(row.get("count", 0), 0))
        patient_count = max(0, _safe_int(row.get("patient_count", 0), 0))
        conf = _safe_float(row.get("confidence", 0), 0.0)
        store[cand] = {
            "candidate_name": cand,
            "count": count,
            "contexts": set(_split_contexts(row.get("contexts", ""))),
            "trigger_labels": set(_split_pipe(row.get("trigger_labels", ""))),
            "source_doc_types": set(_split_pipe(row.get("source_doc_types", ""))),
            "patient_names": set(f"__loaded_patient_{i}" for i in range(patient_count)),
            "example_paths": set(_split_examples(row.get("example_paths", ""))),
            "confidences": ([conf] * count) if count > 0 else [],
        }
    return store


def _load_processed_meta(path: Path) -> Dict[str, dict]:
    rows = safe_read_csv(path)
    out: Dict[str, dict] = {}
    for row in rows:
        key = _patient_key(row.get("glaucoma_type", ""), row.get("patient_name", ""))
        if not key or key == "::":
            continue
        out[key] = {
            "glaucoma_type": normalize_spaces(row.get("glaucoma_type", "")),
            "patient_name": normalize_spaces(row.get("patient_name", "")),
            "patient_id": normalize_spaces(row.get("patient_id", "")),
            "processed_at": normalize_spaces(row.get("processed_at", "")),
        }
    return out


def _write_processed_meta(path: Path, processed_meta: Dict[str, dict]) -> None:
    rows = sorted(processed_meta.values(), key=lambda x: (x["glaucoma_type"], x["patient_name"]))
    write_csv(path, rows, PROCESSED_FIELDS)


def _load_runtime_state(path: Path, logger) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict], dict]:
    if not path.exists():
        return {}, {}, {}, {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"状态文件读取失败，忽略并重建: {path} | {e}")
        return {}, {}, {}, {}
    doctor_store = _deserialize_store(obj.get("doctor_store", {}))
    hospital_store = _deserialize_store(obj.get("hospital_store", {}))
    processed_list = obj.get("processed_patients", [])
    processed_meta: Dict[str, dict] = {}
    if isinstance(processed_list, list):
        for x in processed_list:
            if not isinstance(x, dict):
                continue
            key = _patient_key(x.get("glaucoma_type", ""), x.get("patient_name", ""))
            if key == "::":
                continue
            processed_meta[key] = {
                "glaucoma_type": normalize_spaces(x.get("glaucoma_type", "")),
                "patient_name": normalize_spaces(x.get("patient_name", "")),
                "patient_id": normalize_spaces(x.get("patient_id", "")),
                "processed_at": normalize_spaces(x.get("processed_at", "")),
            }
    stats = obj.get("stats", {}) if isinstance(obj.get("stats", {}), dict) else {}
    logger.info(
        f"loaded runtime state | processed={len(processed_meta)} doctor_store={len(doctor_store)} hospital_store={len(hospital_store)}"
    )
    return doctor_store, hospital_store, processed_meta, stats


def _save_runtime_state(path: Path, doctor_store: Dict[str, dict], hospital_store: Dict[str, dict],
                        processed_meta: Dict[str, dict], stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "doctor_store": _serialize_store(doctor_store),
        "hospital_store": _serialize_store(hospital_store),
        "processed_patients": sorted(processed_meta.values(), key=lambda x: (x["glaucoma_type"], x["patient_name"])),
        "stats": stats,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _flush_outputs(output_root: Path, patient_rows: List[dict], doctor_store: Dict[str, dict], hospital_store: Dict[str, dict],
                   patient_hits: List[dict], processed_meta: Dict[str, dict], state_stats: dict) -> None:
    doctor_rows = finalize_candidate_rows(doctor_store, "doctor_id")
    hospital_rows = finalize_candidate_rows(hospital_store, "hospital_id")

    write_csv(output_root / "patient_registry.csv", patient_rows, PATIENT_REGISTRY_FIELDS)
    write_csv(output_root / "doctor_candidates.csv", doctor_rows, DOCTOR_CANDIDATE_FIELDS)
    write_csv(output_root / "hospital_candidates.csv", hospital_rows, HOSPITAL_CANDIDATE_FIELDS)
    write_csv(output_root / "patient_sensitive_hits.csv", patient_hits, PATIENT_HIT_FIELDS)
    # 提供与 pass2 的默认输入一致的文件名，便于后续人工审核与流水线衔接。
    write_csv(output_root / "patient_sensitive_info.csv", patient_hits, PATIENT_HIT_FIELDS)
    write_csv(output_root / "doctor_registry_template.csv", doctor_rows, DOCTOR_TEMPLATE_FIELDS)
    write_csv(output_root / "hospital_registry_template.csv", hospital_rows, HOSPITAL_TEMPLATE_FIELDS)
    _write_processed_meta(output_root / "processed_patients.csv", processed_meta)
    _save_runtime_state(output_root / "pass1_runtime_state.json", doctor_store, hospital_store, processed_meta, state_stats)


def run(args):
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "pass1_scan_candidates.log")

    patients = discover_patients(input_root)
    if args.max_patients is not None and args.max_patients > 0:
        logger.info(f"🚧 开启测试模式：原数据集共有 {len(patients)} 名患者，当前限制只处理前 {args.max_patients} 名。")
        patients = patients[:args.max_patients]

    patient_rows = [{
        "patient_name": item["patient_name"],
        "patient_id": item["patient_id"],
        "glaucoma_type": item["glaucoma_type"],
        "patient_dir": str(item["patient_dir"]),
    } for item in patients]

    doctor_store: Dict[str, dict] = {}
    hospital_store: Dict[str, dict] = {}
    processed_meta: Dict[str, dict] = {}
    patient_hits: List[dict] = []
    patient_hit_keys = set()
    state_stats = {
        "total_files": 0,
        "total_followup_images": 0,
        "total_patients_discovered": len(patients),
    }

    if args.resume:
        state_path = output_root / "pass1_runtime_state.json"
        doctor_store, hospital_store, processed_meta, loaded_stats = _load_runtime_state(state_path, logger)
        if loaded_stats:
            state_stats.update({
                "total_files": _safe_int(loaded_stats.get("total_files", 0), 0),
                "total_followup_images": _safe_int(loaded_stats.get("total_followup_images", 0), 0),
                "total_patients_discovered": len(patients),
            })

        if not processed_meta and (output_root / "processed_patients.csv").exists():
            processed_meta = _load_processed_meta(output_root / "processed_patients.csv")
            logger.info(f"loaded processed list from csv fallback | processed={len(processed_meta)}")

        if not doctor_store and (output_root / "doctor_candidates.csv").exists():
            doctor_store = _candidate_store_from_rows(safe_read_csv(output_root / "doctor_candidates.csv"))
            logger.warning(f"runtime doctor store rebuilt from doctor_candidates.csv | entries={len(doctor_store)}")
        if not hospital_store and (output_root / "hospital_candidates.csv").exists():
            hospital_store = _candidate_store_from_rows(safe_read_csv(output_root / "hospital_candidates.csv"))
            logger.warning(f"runtime hospital store rebuilt from hospital_candidates.csv | entries={len(hospital_store)}")

        existing_hits = safe_read_csv(output_root / "patient_sensitive_hits.csv")
        if existing_hits:
            existing_hits, _ = clean_patient_sensitive_hits(existing_hits)
            existing_hits = dedupe_hits_globally(existing_hits)
            patient_hits.extend(existing_hits)
            patient_hit_keys.update(build_hit_dedupe_key(x) for x in patient_hits)
            logger.info(f"loaded existing patient hits for resume | rows={len(patient_hits)}")

    # 无论是否 resume，先把 patient_registry 落盘，保证可见进度。
    write_csv(output_root / "patient_registry.csv", patient_rows, PATIENT_REGISTRY_FIELDS)

    ocr_engine = None if args.skip_ocr else get_ocr_engine()
    processed_now = 0
    skipped_done = 0

    for item in patients:
        patient_name = item["patient_name"]
        patient_id = item["patient_id"]
        glaucoma_type = item["glaucoma_type"]
        pdir = item["patient_dir"]
        pkey = _patient_key(glaucoma_type, patient_name)

        if args.resume and pkey in processed_meta:
            skipped_done += 1
            logger.info(f"[{patient_name}] skip | already processed ({processed_meta[pkey].get('processed_at', '')})")
            continue

        logger.info(f"[{patient_name}] scanning | type={glaucoma_type} | id={patient_id}")
        local_hits: List[dict] = []
        local_files = 0
        local_followup_images = 0

        try:
            for docx_path in sorted(
                [p for p in pdir.iterdir() if p.is_file() and p.suffix.lower() == ".docx" and not p.name.startswith("~$")],
                key=lambda p: p.name,
            ):
                state_stats["total_files"] += 1
                local_files += 1
                doc_type = infer_doc_type(docx_path)
                if doc_type in {"admission", "discharge"}:
                    blocks = document_to_blocks(docx_path)
                    block_idx = 0
                    for section_name in ("header", "body", "footer"):
                        for block in blocks[section_name]:
                            block_idx += 1
                            source_path = f"{docx_path}|{block.location}"
                            scan_doctor_candidates(block.text, doc_type, patient_name, source_path, doctor_store)
                            scan_hospital_candidates(block.text, doc_type, patient_name, source_path, hospital_store)
                            local_hits.extend(
                                scan_patient_sensitive_hits(
                                    block.text,
                                    patient_name,
                                    patient_id,
                                    glaucoma_type,
                                    str(docx_path),
                                    doc_type,
                                    block_idx,
                                )
                            )
                elif "术后" in docx_path.stem:
                    images = extract_docx_images_in_order(docx_path)
                    for img_idx, (_, image_bytes) in enumerate(images):
                        state_stats["total_followup_images"] += 1
                        local_followup_images += 1
                        ocr_text = ocr_image_bytes(image_bytes, ocr_engine) if ocr_engine else ""
                        source_path = f"{docx_path}|image[{img_idx}]"
                        scan_doctor_candidates(ocr_text, "followup_ocr", patient_name, source_path, doctor_store)
                        scan_hospital_candidates(ocr_text, "followup_ocr", patient_name, source_path, hospital_store)
                        local_hits.extend(
                            scan_patient_sensitive_hits(
                                ocr_text,
                                patient_name,
                                patient_id,
                                glaucoma_type,
                                str(docx_path),
                                "followup_ocr",
                                img_idx,
                            )
                        )

            cleaned_hits, clean_stats = clean_patient_sensitive_hits(local_hits)
            added_hits = 0
            skipped_dup_hits = 0
            for row in cleaned_hits:
                key = build_hit_dedupe_key(row)
                if key in patient_hit_keys:
                    skipped_dup_hits += 1
                    continue
                patient_hit_keys.add(key)
                patient_hits.append(row)
                added_hits += 1

            processed_meta[pkey] = {
                "glaucoma_type": glaucoma_type,
                "patient_name": patient_name,
                "patient_id": patient_id,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            processed_now += 1

            _flush_outputs(output_root, patient_rows, doctor_store, hospital_store, patient_hits, processed_meta, state_stats)
            logger.info(
                f"[{patient_name}] persisted | files={local_files} followup_images={local_followup_images} "
                f"hits_raw={len(local_hits)} hits_clean={clean_stats['kept_rows']} hits_added={added_hits} "
                f"hits_dup_skip={skipped_dup_hits} dropped_label={clean_stats['dropped_label_rows']} "
                f"dropped_empty={clean_stats['dropped_empty_rows']} truncated={clean_stats['truncated_rows']}"
            )
        except Exception as e:
            logger.exception(f"[{patient_name}] failed | error={e}")
            if args.fail_fast:
                raise
            logger.warning(f"[{patient_name}] continue to next patient due to --fail-fast disabled")

    # 即使本轮没有新增，也做一次最终落盘，确保文件一致。
    _flush_outputs(output_root, patient_rows, doctor_store, hospital_store, patient_hits, processed_meta, state_stats)
    logger.info(
        f"done | total_patients={len(patients)} processed_now={processed_now} skipped_already_done={skipped_done} "
        f"processed_total={len(processed_meta)} files={state_stats['total_files']} followup_images={state_stats['total_followup_images']} "
        f"doctor_candidates={len(doctor_store)} hospital_candidates={len(hospital_store)} patient_hits={len(patient_hits)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pass 1: 扫描医生/医院候选词与患者敏感信息，不做替换。")
    parser.add_argument("--input-root", default=r"E:\0_dataset\glaucoma_icl\glaucoma-ehr")
    parser.add_argument("--output-dir", default=r"D:\AgentSpace\codex\med-agent-4-ophth\data\input\glaucoma\pass1")
    parser.add_argument("--skip-ocr", action="store_true", help="跳过术后 docx 图片 OCR")
    parser.add_argument("--max-patients", type=int, default=None, help="最多处理多少个患者（用于快速测试，例如输入 10）")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="从已处理患者列表继续执行（默认开启）")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="不读取历史进度，重新开始扫描")
    parser.add_argument("--fail-fast", action="store_true", help="遇到单个患者错误时立即中断（默认：记录错误并继续）")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
