
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import re

from med_deid_common import (
    build_patient_rules_from_hits,
    deidentify_text,
    discover_patients,
    document_to_blocks,
    extract_docx_images_in_order,
    get_ocr_engine,
    infer_doc_type,
    load_registry_map,
    normalize_spaces,
    normalize_spaces_keep_tabs,
    ocr_image_bytes,
    safe_read_csv,
    setup_logger,
    write_csv,
    truthy,
)


def find_output_dir(base_output_root: Path, glaucoma_type: str, patient_id: str) -> Path:
    return base_output_root / glaucoma_type / patient_id


NESTED_TABLE_RE = re.compile(r"<<NESTED_TABLE>>\s*(.*?)\s*<</NESTED_TABLE>>", re.DOTALL)


def _render_generic_table(table_text: str, indent: str = "") -> List[str]:
    lines: List[str] = []
    for line in table_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        cols = [normalize_spaces(x) for x in line.split("\t")]
        cols = [x for x in cols if x]
        if not cols:
            continue
        if len(cols) == 1:
            lines.append(f"{indent}{cols[0]}")
        else:
            lines.append(f"{indent}" + " | ".join(cols))
    return lines


def _render_nested_table_for_nlp(table_text: str) -> List[str]:
    def _cell_fix(v: str) -> str:
        return normalize_spaces(v.replace(r"\n", " / "))

    rows = []
    for line in table_text.split("\n"):
        raw = line.strip()
        if not raw:
            continue
        cols = [_cell_fix(x) for x in raw.split("\t")]
        if any(cols):
            rows.append(cols)
    if not rows:
        return []

    header = rows[0]
    # 常见眼科结构：检查项 | 右眼 | 左眼
    if len(header) >= 3 and ("右眼" in header[1]) and ("左眼" in header[2]):
        out = ["[嵌套表格] 列=检查项 | 右眼 | 左眼"]
        for row in rows[1:]:
            item = row[0] if len(row) > 0 else ""
            od = row[1] if len(row) > 1 else ""
            os = row[2] if len(row) > 2 else ""
            item = item or "未命名检查项"
            out.append(f"- {item} | 右眼: {od} | 左眼: {os}")
        return out

    out = ["[嵌套表格]"]
    out.extend(_render_generic_table(table_text, indent=""))
    return out


def _format_cell_for_nlp(cell_text: str) -> str:
    text = normalize_spaces_keep_tabs(cell_text)
    if not text:
        return ""
    parts: List[str] = []
    cursor = 0
    for m in NESTED_TABLE_RE.finditer(text):
        prefix = text[cursor:m.start()].strip()
        if prefix:
            parts.extend(_render_generic_table(prefix))
        nested = m.group(1)
        parts.extend(_render_nested_table_for_nlp(nested))
        cursor = m.end()
    tail = text[cursor:].strip()
    if tail:
        parts.extend(_render_generic_table(tail))
    return "\n".join(parts).strip()


def _rows_to_text(rows: List[List[str]]) -> str:
    lines: List[str] = []
    for ridx, row in enumerate(rows, start=1):
        lines.append(f"[ROW {ridx}]")
        for cidx, cell in enumerate(row, start=1):
            formatted = _format_cell_for_nlp(cell)
            if not formatted:
                continue
            cell_lines = formatted.split("\n")
            if len(cell_lines) == 1:
                lines.append(f"C{cidx}: {cell_lines[0]}")
            else:
                lines.append(f"C{cidx}:")
                lines.extend(f"  {x}" for x in cell_lines)
    return "\n".join(lines).strip()


def write_plain_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 带 BOM 的 UTF-8，兼容 Windows 记事本/Excel 直接打开中文。
    path.write_text(text, encoding="utf-8-sig")


def structure_to_plain_text(structure: dict) -> str:
    chunks: List[str] = []
    for section_name in ("header", "body", "footer"):
        section_blocks = structure.get(section_name, [])
        if not section_blocks:
            continue
        chunks.append(f"### {section_name.upper()} ###")
        for block in section_blocks:
            chunks.append(f"[{block.location}]")
            if block.kind == "paragraph":
                if block.text:
                    chunks.append(normalize_spaces(block.text))
            else:
                rows = block.rows or []
                if rows:
                    chunks.append(_rows_to_text(rows))
                elif block.text:
                    chunks.append(_format_cell_for_nlp(block.text))
        chunks.append("")
    return "\n".join(chunks).strip() + "\n"


def transform_structure(structure: dict, patient_name: str, patient_id: str, patient_rules: dict,
                        doctor_map: dict, hospital_map: dict) -> dict:
    out = {"header": [], "body": [], "footer": []}
    for section_name in ("header", "body", "footer"):
        for block in structure[section_name]:
            if block.kind == "paragraph":
                new_text = deidentify_text(block.text, patient_name, patient_id, patient_rules, doctor_map, hospital_map)
                out[section_name].append(block.__class__(location=block.location, kind=block.kind, text=new_text))
            else:
                new_rows = []
                for row in block.rows or []:
                    new_rows.append([deidentify_text(cell, patient_name, patient_id, patient_rules, doctor_map, hospital_map) for cell in row])
                new_text = _rows_to_text(new_rows)
                out[section_name].append(block.__class__(location=block.location, kind=block.kind, text=new_text, rows=new_rows))
    return out


def _load_patient_sensitive_rows(path_candidates: List[Path], logger) -> List[dict]:
    for p in path_candidates:
        rows = safe_read_csv(p)
        if rows:
            logger.info(f"patient sensitive source: {p} | rows={len(rows)}")
            return rows
    existing = [str(p) for p in path_candidates if p.exists()]
    raise FileNotFoundError(
        f"未找到可用的 patient_sensitive_info.csv。已检查: {[str(p) for p in path_candidates]}；存在但为空: {existing}"
    )


def _load_registry_map_relaxed(path: Path, key_field: str, id_field: str, logger, label: str) -> Dict[str, str]:
    strict_map = load_registry_map(path, key_field, id_field)
    rows = safe_read_csv(path)
    if strict_map:
        logger.info(f"{label} map loaded (confirmed=1): {len(strict_map)} entries from {path}")
        return strict_map

    # 常见误配置：id 已填写，但 confirmed 仍是 0，导致映射为空并且完全不替换。
    fallback_map: Dict[str, str] = {}
    id_filled = 0
    confirmed_falsy = 0
    for row in rows:
        key = normalize_spaces(row.get(key_field, ""))
        val = normalize_spaces(row.get(id_field, ""))
        if not key or not val:
            continue
        id_filled += 1
        if not truthy(row.get("confirmed", "1")):
            confirmed_falsy += 1
        fallback_map[key] = val

    if fallback_map:
        logger.warning(
            f"{label} map fallback enabled: strict map is empty, but found {id_filled} rows with non-empty {id_field}. "
            f"{confirmed_falsy} of them are confirmed=0/false. Fallback will still apply replacements."
        )
    else:
        logger.warning(f"{label} map is empty: no usable rows in {path}")
    return fallback_map


def _count_source_hits(text: str, mapping: Dict[str, str]) -> int:
    if not text or not mapping:
        return 0
    return sum(1 for src in mapping.keys() if src and src in text)


def run(args):
    input_root = Path(args.input_root).resolve()
    pass1_dir = Path(args.pass1_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "pass2_deidentify.log")

    patient_registry_path_candidates = [
        Path(args.patient_registry).resolve(),
        pass1_dir / "patient_registry.csv",
    ]
    patient_registry = []
    patient_registry_path = patient_registry_path_candidates[0]
    for p in patient_registry_path_candidates:
        rows = safe_read_csv(p)
        if rows:
            patient_registry = rows
            patient_registry_path = p
            break
    if not patient_registry:
        raise FileNotFoundError(f"缺少 patient_registry.csv，可选路径: {[str(p) for p in patient_registry_path_candidates]}")
    logger.info(f"patient registry source: {patient_registry_path} | rows={len(patient_registry)}")

    patient_hits = _load_patient_sensitive_rows([
        Path(args.patient_sensitive_info).resolve(),
        pass1_dir / "patient_sensitive_info.csv",
    ], logger)
    patient_rules = build_patient_rules_from_hits(patient_hits, patient_registry)
    doctor_map = _load_registry_map_relaxed(Path(args.doctor_registry), "candidate_name", "doctor_id", logger, "doctor")
    hospital_map = _load_registry_map_relaxed(Path(args.hospital_registry), "candidate_name", "hospital_id", logger, "hospital")
    if args.skip_ocr:
        ocr_engine = None
    else:
        try:
            ocr_engine = get_ocr_engine()
        except Exception as e:
            logger.warning(f"OCR engine unavailable, fallback to skip OCR for follow-up images: {e}")
            ocr_engine = None

    patient_id_map = {row["patient_name"]: row["patient_id"] for row in patient_registry}
    manifest_rows = []
    counts = defaultdict(int)

    patients = discover_patients(input_root)
    if args.max_patients is not None and args.max_patients > 0:
        logger.info(f"🚧 test mode: discovered={len(patients)} only process first {args.max_patients} patients")
        patients = patients[:args.max_patients]

    for item in patients:
        patient_name = item["patient_name"]
        glaucoma_type = item["glaucoma_type"]
        pdir = item["patient_dir"]
        patient_id = patient_id_map.get(patient_name)
        if not patient_id:
            logger.warning(f"skip patient without id: {patient_name}")
            continue
        out_dir = find_output_dir(output_root, glaucoma_type, patient_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        rules = patient_rules.get(patient_name, {"patient_id": patient_id, "replace_patient_terms": {patient_name}, "delete_values": set()})

        admission_idx = 0
        discharge_idx = 0
        for file in sorted(
            [p for p in pdir.iterdir() if p.is_file() and p.suffix.lower() == ".docx" and not p.name.startswith("~$")],
            key=lambda p: p.name,
        ):
            doc_type = infer_doc_type(file)
            if doc_type == "admission":
                counts["source_text_docx"] += 1
                structure = document_to_blocks(file)
                original_text = structure_to_plain_text(structure)
                new_structure = transform_structure(structure, patient_name, patient_id, rules, doctor_map, hospital_map)
                deid_text = structure_to_plain_text(new_structure)
                output_path = out_dir / f"{patient_id}_admission_{admission_idx}.txt"
                write_plain_text_file(output_path, deid_text)
                counts["doctor_hits_in_source"] += _count_source_hits(original_text, doctor_map)
                counts["hospital_hits_in_source"] += _count_source_hits(original_text, hospital_map)
                counts["doctor_ids_in_output"] += sum(deid_text.count(x) for x in doctor_map.values())
                counts["hospital_ids_in_output"] += sum(deid_text.count(x) for x in hospital_map.values())
                manifest_rows.append({"patient_name": patient_name, "patient_id": patient_id, "glaucoma_type": glaucoma_type, "source_file": str(file), "source_doc_type": "admission", "record_index": admission_idx, "output_file": str(output_path)})
                admission_idx += 1
            elif doc_type == "discharge":
                counts["source_text_docx"] += 1
                structure = document_to_blocks(file)
                original_text = structure_to_plain_text(structure)
                new_structure = transform_structure(structure, patient_name, patient_id, rules, doctor_map, hospital_map)
                deid_text = structure_to_plain_text(new_structure)
                output_path = out_dir / f"{patient_id}_discharge_{discharge_idx}.txt"
                write_plain_text_file(output_path, deid_text)
                counts["doctor_hits_in_source"] += _count_source_hits(original_text, doctor_map)
                counts["hospital_hits_in_source"] += _count_source_hits(original_text, hospital_map)
                counts["doctor_ids_in_output"] += sum(deid_text.count(x) for x in doctor_map.values())
                counts["hospital_ids_in_output"] += sum(deid_text.count(x) for x in hospital_map.values())
                manifest_rows.append({"patient_name": patient_name, "patient_id": patient_id, "glaucoma_type": glaucoma_type, "source_file": str(file), "source_doc_type": "discharge", "record_index": discharge_idx, "output_file": str(output_path)})
                discharge_idx += 1
            elif "术后" in file.stem:
                images = extract_docx_images_in_order(file)
                followup_idx = 0
                for img_idx, (img_name, img_bytes) in enumerate(images):
                    counts["source_followup_images"] += 1
                    ocr_text = ocr_image_bytes(img_bytes, ocr_engine) if ocr_engine else ""
                    deid_text = deidentify_text(ocr_text, patient_name, patient_id, rules, doctor_map, hospital_map)
                    output_path = out_dir / f"{patient_id}_followup_{followup_idx}.txt"
                    metadata = [f"patient_id: {patient_id}", f"glaucoma_type: {glaucoma_type}", f"source_docx: {file.name}", f"source_image: {img_name}", f"ocr_record_index: {img_idx}"]
                    text_payload = "\n".join(metadata + ["", deid_text]).strip() + "\n"
                    write_plain_text_file(output_path, text_payload)
                    counts["doctor_hits_in_source"] += _count_source_hits(ocr_text, doctor_map)
                    counts["hospital_hits_in_source"] += _count_source_hits(ocr_text, hospital_map)
                    counts["doctor_ids_in_output"] += sum(deid_text.count(x) for x in doctor_map.values())
                    counts["hospital_ids_in_output"] += sum(deid_text.count(x) for x in hospital_map.values())
                    manifest_rows.append({"patient_name": patient_name, "patient_id": patient_id, "glaucoma_type": glaucoma_type, "source_file": str(file), "source_doc_type": "followup_ocr", "record_index": followup_idx, "output_file": str(output_path)})
                    followup_idx += 1
        logger.info(f"[{patient_name}] done | patient_id={patient_id} | out_dir={out_dir}")

    write_csv(output_root / "deidentify_manifest.csv", manifest_rows, ["patient_name", "patient_id", "glaucoma_type", "source_file", "source_doc_type", "record_index", "output_file"])
    logger.info(
        "done | text_docx=%s followup_images=%s outputs=%s doctor_source_hits=%s hospital_source_hits=%s "
        "doctor_ids_in_output=%s hospital_ids_in_output=%s",
        counts["source_text_docx"],
        counts["source_followup_images"],
        len(manifest_rows),
        counts["doctor_hits_in_source"],
        counts["hospital_hits_in_source"],
        counts["doctor_ids_in_output"],
        counts["hospital_ids_in_output"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pass 2: 基于确认后的 registry 执行正式匿名化。")
    parser.add_argument("--input-root", default=r"E:\0_dataset\glaucoma_icl\glaucoma-ehr")
    parser.add_argument("--pass1-dir", default=r"E:\0_dataset\glaucoma_icl\glaucoma-117-case-to-be-anonymized")
    parser.add_argument("--patient-registry", default=r"E:\0_dataset\glaucoma_icl\glaucoma-117-case-to-be-anonymized\patient_registry.csv")
    parser.add_argument("--patient-sensitive-info", default=r"E:\0_dataset\glaucoma_icl\glaucoma-117-case-to-be-anonymized\patient_sensitive_info.csv")
    parser.add_argument("--doctor-registry", default=r"E:\0_dataset\glaucoma_icl\glaucoma-117-case-to-be-anonymized\doctor_registry.csv")
    parser.add_argument("--hospital-registry", default=r"E:\0_dataset\glaucoma_icl\glaucoma-117-case-to-be-anonymized\hospital_registry.csv")
    parser.add_argument("--output-root", default=r"D:\AgentSpace\codex\med-agent-4-ophth\data\input\glaucoma-ehr-deidentified")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--max-patients", type=int, default=None, help="最多处理多少个患者（用于快速测试）")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
