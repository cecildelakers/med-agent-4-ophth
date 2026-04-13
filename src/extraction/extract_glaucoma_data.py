#!/usr/bin/env python3
"""
Extract one glaucoma patient from de-identified TXT EHR files to intermediate schema JSON,
using a local HuggingFace causal LLM (e.g., Qwen3-4B on NUS HPC).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RECORD_RE = re.compile(r"^(PT_[0-9]{6})_(admission|discharge|followup)_([0-9]+)\.txt$", re.I)
ID_PATTERNS = {
    "patient": re.compile(r"^PT_[0-9]{6}$"),
    "doctor": re.compile(r"^DR_[0-9]{6}$"),
    "hospital": re.compile(r"^HP_[0-9]{6}$"),
    "record": re.compile(r"^DOC_[0-9]{6}$"),
    "episode": re.compile(r"^EP_[0-9]{6}$"),
    "followup": re.compile(r"^FU_[0-9]{6}$"),
    "evidence": re.compile(r"^EV_[0-9]{6}$"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Glaucoma EHR extraction")
    p.add_argument("--patient-dir", required=True)
    p.add_argument("--schema-path", default="data/input/intermediate_schema.json")
    p.add_argument("--schema-hint-path", default="data/input/intermediate_schema_hint.json")
    p.add_argument("--prompt-template", default="prompts/glaucoma_ehr_extraction_instruction_zh.txt")
    p.add_argument("--model-path", default="/scratch/e1538612/models/Qwen3-4B")
    p.add_argument("--output-path", required=True)
    p.add_argument("--ground-truth", default="")
    p.add_argument("--eval-output", default="")
    p.add_argument("--save-raw-response", default="")
    p.add_argument("--save-rendered-prompt", default="")
    p.add_argument("--max-record-chars", type=int, default=14000)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dry-run-no-llm", action="store_true")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-reduce-record-chars", type=int, default=2000)
    p.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    p.add_argument("--prompt-total-record-chars", type=int, default=18000)
    return p.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(read_text(path))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_upper(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest().upper()


def clean_ehr_text(text: str) -> str:
    skip_prefixes = (
        "### HEADER ###",
        "### FOOTER ###",
        "[header",
        "[footer",
        "[body.",
        "[ROW ",
        "DCSoft inside",
    )
    cleaned: List[str] = []
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.replace("\u00a0", " ").strip()
        if not line:
            continue
        if line.startswith(skip_prefixes):
            continue
        line = re.sub(r"\s+", " ", line)
        cleaned.append(line)
    return "\n".join(cleaned)


def normalize_date(value: str) -> str:
    value = value.strip()
    m = re.search(r"([0-9]{4})[-./]([0-9]{1,2})[-./]([0-9]{1,2})", value)
    if m:
        y, mm, dd = m.groups()
        return f"{int(y):04d}-{int(mm):02d}-{int(dd):02d}"
    m = re.search(r"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})(?:日)?", value)
    if m:
        y, mm, dd = m.groups()
        return f"{int(y):04d}-{int(mm):02d}-{int(dd):02d}"
    return ""


def extract_doc_date(record_type: str, text: str) -> str:
    patterns = []
    if record_type == "admission":
        patterns = [r"入院时间[:：]\s*([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})", r"记录时间[:：]\s*([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})"]
    elif record_type == "discharge":
        patterns = [r"出院日期[:：]\s*([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})", r"入院日期[:：]\s*([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})"]
    elif record_type == "followup":
        patterns = [r"就诊日期[:：]\s*([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})"]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return normalize_date(m.group(1))
    m = re.search(r"([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})", text)
    if m:
        return normalize_date(m.group(1))
    m = re.search(r"([0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日)", text)
    if m:
        return normalize_date(m.group(1))
    return ""


def build_record_id(patient_id: str, record_type: str, idx: int) -> str:
    pnum = int(patient_id.split("_")[1])
    if record_type == "admission":
        slot = 0 + idx
    elif record_type == "discharge":
        slot = 1 + idx
    else:
        slot = 10 + idx
    return f"DOC_{pnum:03d}{slot:03d}"


def discover_records(patient_dir: Path, max_record_chars: int) -> Tuple[str, str, List[Dict[str, Any]]]:
    items: List[Tuple[Path, str, int, str]] = []
    for path in patient_dir.glob("*.txt"):
        m = RECORD_RE.match(path.name)
        if not m:
            continue
        pid, rtype, idx = m.groups()
        items.append((path, rtype.lower(), int(idx), pid))
    if not items:
        raise RuntimeError(f"No matching txt records in {patient_dir}")

    type_rank = {"admission": 0, "discharge": 1, "followup": 2}
    items.sort(key=lambda x: (type_rank.get(x[1], 99), x[2], x[0].name.lower()))

    pids = sorted({x[3] for x in items})
    if len(pids) != 1:
        raise RuntimeError(f"Expected one patient id, got: {pids}")
    patient_id = pids[0]

    records: List[Dict[str, Any]] = []
    for path, rtype, idx, _ in items:
        raw = read_text(path)
        cleaned = clean_ehr_text(raw)
        records.append(
            {
                "record_id": build_record_id(patient_id, rtype, idx),
                "record_type": rtype,
                "record_index": idx,
                "source_path": path.as_posix(),
                "document_date": extract_doc_date(rtype, raw),
                "hash_sha256": sha256_upper(raw),
                "text": cleaned[:max_record_chars] if len(cleaned) > max_record_chars else cleaned,
                "_raw_text": raw,
                "_clean_text": cleaned,
            }
        )

    glaucoma_type = patient_dir.parent.name.strip().upper()
    if glaucoma_type not in {"AACG", "CACG", "POAG"}:
        glaucoma_type = "Unknown"
    return patient_id, glaucoma_type, records


def build_compact_prompt_schema(
    schema_json: Dict[str, Any],
    hint_json: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    episode_tpl = schema_json["patients"][0]["episodes"][0]
    eye_fields = list(episode_tpl["s1_pre_intervention_state"]["eyes_data"]["OD"].keys())
    compact_schema = {
        "payload_top_keys": [
            "hospitals",
            "doctors",
            "patient_demographics",
            "patient_status_tracking",
            "episodes",
            "evidence_catalog",
        ],
        "id_rules": schema_json.get("id_rules", {}),
        "episode_keys": list(episode_tpl.keys()),
        "followup_record_keys": list(episode_tpl["follow_up_state"]["followup_records"][0].keys()),
        "eye_fields": eye_fields,
    }
    compact_hint = {
        "glaucoma_type_enum": ["AACG", "CACG", "POAG", "Unknown"],
        "glaucoma_diagnosis_type_enum": ["AACG", "CACG", "POAG", "Unknown"],
        "laterality_enum": ["OD", "OS", "OU", "Unknown"],
        "status_enum": {
            "episode_status": ["active", "closed", "merged", "unknown"],
            "stage_status": ["complete", "partial", "missing"],
            "verification_status": [
                "unverified",
                "machine_checked",
                "clinician_reviewed",
                "finalized",
            ],
        },
        "field_constraints_subset": {
            k: v
            for k, v in hint_json.get("field_constraints", {}).items()
            if "glaucoma_diagnosis_type" in k
            or ".a_treatment_action.medications" in k
            or ".current_medications" in k
            or k.endswith(".age_first_seen")
            or k.endswith(".biological_sex")
            or k.endswith(".race")
            or k.endswith(".ethnicity")
            or k.endswith(".observation_date")
            or k.endswith(".followup_date")
            or k.endswith(".episode_status")
            or k.endswith(".verifiable")
            or k.endswith(".ucva")
            or k.endswith(".bcva")
            or k.endswith(".iop")
            or k.endswith(".admission_date")
            or k.endswith(".discharge_date")
        },
        "field_definitions_subset": {
            k: v
            for k, v in hint_json.get("field_definitions", {}).items()
            if k.endswith(".ucva")
            or k.endswith(".bcva")
            or k.endswith(".iop")
            or ".a_treatment_action.medications" in k
            or ".current_medications" in k
            or k.endswith(".age_first_seen")
            or k.endswith(".biological_sex")
            or k.endswith(".race")
            or k.endswith(".ethnicity")
            or k.endswith(".followup_date")
            or k.endswith(".admission_date")
            or k.endswith(".discharge_date")
            or "glaucoma_diagnosis_type" in k
        },
        "extraction_guidelines": hint_json.get("extraction_guidelines", {}),
        "normalization_rules": hint_json.get("normalization_rules", {}),
    }
    return compact_schema, compact_hint


def parse_float(text: str) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(text)
    except Exception:  # pylint: disable=broad-except
        return None


VISUAL_VALUE_RE = re.compile(
    r"(手动/[0-9]+(?:cm|m)?|指数/[0-9]+(?:cm|m)?|光感|无光感|[0-9]+(?:\.[0-9]+)?-?)"
)


def normalize_visual_token(value: str) -> str:
    token = (value or "").strip().strip("，,。；;:：")
    token = token.replace(" ", "")
    if not token:
        return ""
    return token


def normalize_eye_text(text: str) -> str:
    out = text
    out = out.replace("0D", "OD").replace("O0D", "OD")
    out = out.replace("0S", "OS").replace("O0S", "OS")
    out = out.replace("右眼", "OD").replace("左眼", "OS")
    out = re.sub(r"\s+", " ", out)
    return out


def parse_visual_pair(segment: str) -> Tuple[str, str]:
    seg = segment.strip().strip("。；;")
    if not seg:
        return "", ""

    seg = normalize_eye_text(seg).replace("欠配合", "")
    seg = re.sub(r"\s+", "", seg)

    # explicit style: 裸眼:0.4，矫正视力:0.5
    m_uc = re.search(r"裸眼[:：]?([^，,；;。]+)", seg)
    ucva = ""
    bcva = ""
    if m_uc:
        mv = VISUAL_VALUE_RE.search(m_uc.group(1))
        if mv:
            ucva = normalize_visual_token(mv.group(1))
    m_bc = re.search(r"(?:矫正视力|戴镜视力)[:：]?([^，,；;。]+)", seg)
    if m_bc:
        mv = VISUAL_VALUE_RE.search(m_bc.group(1))
        if mv:
            bcva = normalize_visual_token(mv.group(1))
    if ucva or bcva:
        return ucva, bcva

    # compact style: 0.4-戴镜：。 / 0.5戴镜：0.5
    m_pair = re.search(
        r"(?P<uc>(?:手动/[0-9]+(?:cm|m)?|指数/[0-9]+(?:cm|m)?|光感|无光感|[0-9]+(?:\.[0-9]+)?-?))"
        r"戴镜[:：]?(?P<bc>(?:手动/[0-9]+(?:cm|m)?|指数/[0-9]+(?:cm|m)?|光感|无光感|[0-9]+(?:\.[0-9]+)?-?))?",
        seg,
    )
    if m_pair:
        ucva = normalize_visual_token(m_pair.group("uc") or "")
        bcva = normalize_visual_token(m_pair.group("bc") or "")
        return ucva, bcva

    # fallback: first visual token as UCVA
    m_simple = VISUAL_VALUE_RE.search(seg)
    if m_simple:
        return normalize_visual_token(m_simple.group(1)), ""
    return "", ""


def extract_followup_regex_facts(raw_text: str) -> Dict[str, Any]:
    raw = raw_text
    text = normalize_eye_text(raw_text)
    out: Dict[str, Any] = {
        "followup_date": "",
        "OD": {"ucva": "", "bcva": "", "iop": None, "cup_to_disc_ratio": None},
        "OS": {"ucva": "", "bcva": "", "iop": None, "cup_to_disc_ratio": None},
    }
    m_date = re.search(r"就诊日期[:：]\s*([0-9]{4}[-./][0-9]{1,2}[-./][0-9]{1,2})", text)
    if m_date:
        out["followup_date"] = normalize_date(m_date.group(1))

    # vision line
    m_vis_line = re.search(r"视力[:：]\s*([^\n]+)", text)
    if m_vis_line:
        vis = m_vis_line.group(1)
        vis = normalize_eye_text(vis)
        od_seg = ""
        os_seg = ""
        m_od = re.search(r"OD[:：]\s*(.*?)(?=OS[:：]|$)", vis)
        m_os = re.search(r"OS[:：]\s*(.*)$", vis)
        if m_od:
            od_seg = m_od.group(1)
        if m_os:
            os_seg = m_os.group(1)
        uc, bc = parse_visual_pair(od_seg)
        out["OD"]["ucva"] = uc
        out["OD"]["bcva"] = bc
        uc, bc = parse_visual_pair(os_seg)
        out["OS"]["ucva"] = uc
        out["OS"]["bcva"] = bc

    # iop line
    m_iop_line = re.search(r"眼压[:：]\s*([^\n]+)", text)
    if m_iop_line:
        iop = normalize_eye_text(m_iop_line.group(1))
        m_od_iop = re.search(r"OD[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*mmHg", iop, re.I)
        m_os_iop = re.search(r"OS[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*mmHg", iop, re.I)
        if m_od_iop:
            out["OD"]["iop"] = parse_float(m_od_iop.group(1))
        if m_os_iop:
            out["OS"]["iop"] = parse_float(m_os_iop.group(1))

    if out["OD"]["iop"] is None or out["OS"]["iop"] is None:
        m_ta_line = re.search(r"TA[:：]\s*([^\n]+)", text, re.I)
        if m_ta_line:
            ta = normalize_eye_text(m_ta_line.group(1))
            if out["OD"]["iop"] is None:
                m_od_ta = re.search(r"OD[:：]?\s*([0-9]+(?:\.[0-9]+)?)", ta, re.I)
                if m_od_ta:
                    out["OD"]["iop"] = parse_float(m_od_ta.group(1))
            if out["OS"]["iop"] is None:
                m_os_ta = re.search(r"OS[:：]?\s*([0-9]+(?:\.[0-9]+)?)", ta, re.I)
                if m_os_ta:
                    out["OS"]["iop"] = parse_float(m_os_ta.group(1))

    # C/D ratio: parse from exam eye blocks first to avoid cross-eye leakage.
    exam_sec = raw
    m_exam = re.search(r"专科检查[:：]([\s\S]{0,2400}?)(?:辅助检查[:：]|初步诊断[:：]|处理[:：]|随访医嘱[:：]|$)", raw)
    if m_exam:
        exam_sec = m_exam.group(1)
    m_od_block = re.search(r"右眼[:：]([\s\S]{0,900}?)(?:左眼[:：]|$)", exam_sec, re.S)
    m_os_block = re.search(r"左眼[:：]([\s\S]{0,900}?)(?:辅助检查[:：]|初步诊断[:：]|处理[:：]|随访医嘱[:：]|$)", exam_sec, re.S)
    if m_od_block:
        m_cd = re.search(r"C\s*/\s*D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", m_od_block.group(1), re.S)
        if m_cd:
            out["OD"]["cup_to_disc_ratio"] = parse_float(m_cd.group(1))
    if m_os_block:
        m_cd = re.search(r"C\s*/\s*D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", m_os_block.group(1), re.S)
        if m_cd:
            out["OS"]["cup_to_disc_ratio"] = parse_float(m_cd.group(1))

    # fallback for mixed OCR
    m_od_cd = re.search(r"右眼[:：][\s\S]{0,420}?C\s*/\s*D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", raw, re.S)
    m_os_cd = re.search(r"左眼[:：][\s\S]{0,420}?C\s*/\s*D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", raw, re.S)
    if m_od_cd and out["OD"]["cup_to_disc_ratio"] is None:
        out["OD"]["cup_to_disc_ratio"] = parse_float(m_od_cd.group(1))
    if m_os_cd and out["OS"]["cup_to_disc_ratio"] is None:
        out["OS"]["cup_to_disc_ratio"] = parse_float(m_os_cd.group(1))
    if out["OD"]["cup_to_disc_ratio"] is None:
        m_od_cd2 = re.search(r"OD[:：][\s\S]{0,260}?C/D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.S)
        if m_od_cd2:
            out["OD"]["cup_to_disc_ratio"] = parse_float(m_od_cd2.group(1))
    if out["OS"]["cup_to_disc_ratio"] is None:
        m_os_cd2 = re.search(r"OS[:：][\s\S]{0,260}?C/D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.S)
        if m_os_cd2:
            out["OS"]["cup_to_disc_ratio"] = parse_float(m_os_cd2.group(1))
    if out["OD"]["cup_to_disc_ratio"] is None and out["OS"]["cup_to_disc_ratio"] is None:
        cds = re.findall(r"C\s*/\s*D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.S)
        if len(cds) >= 2:
            out["OD"]["cup_to_disc_ratio"] = parse_float(cds[0])
            out["OS"]["cup_to_disc_ratio"] = parse_float(cds[1])
        elif len(cds) == 1:
            v = parse_float(cds[0])
            out["OD"]["cup_to_disc_ratio"] = v
            out["OS"]["cup_to_disc_ratio"] = v
    return out


def infer_eye_from_text(text: str, default: str = "Unknown") -> str:
    if "右眼" in text and "左眼" in text:
        return "OU"
    if "右眼" in text:
        return "OD"
    if "左眼" in text:
        return "OS"
    if "OD" in text and "OS" in text:
        return "OU"
    if "OD" in text:
        return "OD"
    if "OS" in text:
        return "OS"
    return default


def normalize_surgery_name(name: str, eye: str) -> str:
    _ = eye
    return re.sub(r"\s+", " ", name).strip()


MED_NAME_RE = re.compile(
    r"([0-9]+%?[\u4e00-\u9fa5A-Za-z0-9\-]{0,22}(?:滴眼液|眼液|眼膏|眼用凝胶|凝胶)|[\u4e00-\u9fa5A-Za-z0-9\-]{2,24}(?:滴眼液|眼液|眼膏|眼用凝胶|凝胶))"
)
MED_KEYWORD_HINTS = (
    "滴眼液",
    "眼液",
    "眼膏",
    "眼用凝胶",
    "凝胶",
    "滴眼",
    "白力特",
    "可乐必妥",
    "托吡卡胺",
    "贝美前列素",
    "布林佐胺",
    "左氧氟沙星",
    "妥布霉素",
    "泼尼松龙",
    "玻璃酸钠",
    "噻吗洛尔",
    "卡替洛尔",
)
MED_GENERIC_STOP_WORDS = {
    "目前用药",
    "双眼",
    "右眼",
    "左眼",
    "每天",
    "每日",
    "每晚",
    "一次",
    "治疗",
    "无",
    "未诉",
    "术眼",
    "滴眼",
    "解释病情",
    "特殊不适",
    "出院",
    "带药",
    "用药",
}
MED_GENERIC_BAD_SUBSTRINGS = (
    "未诉",
    "不适",
    "复查",
    "医嘱",
    "建议",
    "滴右眼",
    "滴左眼",
    "术眼",
    "停用",
    "改每天",
    "每次",
    "次/天",
    "治疗",
    "点眼",
    "点术眼",
)
KNOWN_GENERIC_MED_NAMES = {
    "白力特",
    "可乐必妥",
    "托吡卡胺",
    "贝美前列素",
    "布林佐胺",
    "噻吗洛尔",
    "卡替洛尔",
    "毛果芸香碱",
    "玻璃酸钠",
    "左氧氟沙星",
    "妥布霉素",
    "泼尼松龙",
    "更昔洛韦",
    "卡波姆",
    "阿托品",
}


def normalize_med_name(name: str) -> str:
    out = re.sub(r"\s+", "", name or "")
    out = out.strip("，,；;。:：")
    out = re.sub(r"^(予以|继续|改为|改用|给予|应用|使用|点用|滴用)", "", out)
    out = re.sub(r"(治疗|用药)$", "", out)
    out = out.replace("眼用 凝胶", "眼用凝胶")
    if out.endswith("滴眼"):
        out = out + "液"
    return out


def infer_med_eye(text: str, surgery_eye: str = "Unknown") -> str:
    if "双眼" in text or "OU" in text:
        return "OU"
    if "右眼" in text or "OD" in text:
        return "OD"
    if "左眼" in text or "OS" in text:
        return "OS"
    if "术眼" in text and surgery_eye in {"OD", "OS"}:
        return surgery_eye
    return "Unknown"


def infer_med_purpose(text: str) -> str:
    if "降眼压" in text:
        return "降眼压"
    if "抗炎" in text:
        return "抗炎"
    if "预防感染" in text or "抗感染" in text:
        return "预防感染"
    if "润滑" in text or "干眼" in text:
        return "润滑/干眼治疗"
    return ""


def infer_med_phase(text: str, default_phase: str = "unknown") -> str:
    if "出院带药" in text or "出院用药" in text:
        return "discharge"
    if "术前" in text:
        return "pre_op"
    if "术后" in text:
        return "post_op"
    if "随访" in text or "复查" in text:
        return "followup"
    if "目前用药" in text or "长期用药" in text:
        return "chronic"
    return default_phase


def extract_med_frequency(text: str) -> str:
    patterns = [
        r"(每天[0-9一二三四五六七八九十两]+次)",
        r"(每日[0-9一二三四五六七八九十两]+次)",
        r"(每晚一次)",
        r"(每日一次)",
        r"(每晚[0-9一二三四五六七八九十两]+次)",
        r"(2小时1次/天\*7天(?:\s*改每天[0-9]+次\*7天){0,3})",
        r"(每[0-9]+小时1次)",
        r"\b(qd|bid|tid|qid|q[0-9]+h)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return ""


def looks_like_med_generic_token(token: str) -> bool:
    t = normalize_med_name(token)
    if not t:
        return False
    if t in MED_GENERIC_STOP_WORDS:
        return False
    if len(t) < 2 or len(t) > 18:
        return False
    if any(x in t for x in MED_GENERIC_BAD_SUBSTRINGS):
        return False
    if re.search(r"[0-9/:%*×xX]", t):
        return False
    if not re.fullmatch(r"[\u4e00-\u9fa5A-Za-z\-]+", t):
        return False
    if re.search(r"(滴眼液|眼液|眼膏|眼用凝胶|凝胶|滴眼)$", t):
        return True
    if t in KNOWN_GENERIC_MED_NAMES:
        return True
    return False


def extract_med_names_from_text(text: str, allow_generic: bool = False) -> List[str]:
    names: List[str] = []
    for m in re.finditer(r"[“\"]([^”\"]{1,48})[”\"]", text):
        seg = m.group(1).strip()
        for mm in MED_NAME_RE.findall(seg):
            names.append(normalize_med_name(mm))
    for mm in MED_NAME_RE.findall(text):
        names.append(normalize_med_name(mm))

    if allow_generic:
        for token in re.split(r"[+、，,；;\s]+", text):
            if looks_like_med_generic_token(token):
                names.append(normalize_med_name(token))

    out: List[str] = []
    seen = set()
    for n in names:
        if not n:
            continue
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def build_med_rows_from_block(
    text: str,
    source_record_id: str,
    phase: str,
    surgery_eye: str = "Unknown",
    allow_generic: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    chunks = [x for x in re.split(r"[\n\r]+", text) if x.strip()]
    if not chunks:
        chunks = [text]
    for chunk in chunks:
        names = extract_med_names_from_text(chunk, allow_generic=allow_generic)
        if not names:
            continue
        eye = infer_med_eye(chunk, surgery_eye=surgery_eye)
        freq = extract_med_frequency(chunk)
        purpose = infer_med_purpose(chunk)
        phase_cur = infer_med_phase(chunk, default_phase=phase)
        for name in names:
            rows.append(
                {
                    "action_id": "",
                    "phase": phase_cur,
                    "eye": eye,
                    "name": name,
                    "frequency": freq,
                    "purpose": purpose,
                    "source_record_id": source_record_id,
                }
            )
    return rows


def dedupe_med_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    phase_rank = {"discharge": 6, "post_op": 5, "pre_op": 4, "followup": 3, "chronic": 2, "intra_op": 1, "unknown": 0}
    eye_rank = {"OD": 3, "OS": 3, "OU": 2, "Unknown": 0}

    def row_score(r: Dict[str, Any]) -> Tuple[int, int, int, int]:
        return (
            phase_rank.get(r.get("phase", "unknown"), 0),
            eye_rank.get(r.get("eye", "Unknown"), 0),
            1 if r.get("frequency", "") else 0,
            1 if r.get("purpose", "") else 0,
        )

    out: List[Dict[str, Any]] = []
    for raw in rows:
        r = copy.deepcopy(raw)
        r["name"] = normalize_med_name(r.get("name", ""))
        if not r["name"]:
            continue
        if not re.search(r"(滴眼液|眼液|眼膏|眼用凝胶|凝胶|滴眼)$", r["name"]) and r["name"] not in KNOWN_GENERIC_MED_NAMES:
            continue

        merged = False
        for idx, ex in enumerate(out):
            same_source = ex.get("source_record_id", "") == r.get("source_record_id", "")
            same_name = ex.get("name", "") == r.get("name", "")
            phase_compatible = ex.get("phase", "") == r.get("phase", "") or "unknown" in {ex.get("phase", ""), r.get("phase", "")}
            eye_compatible = ex.get("eye", "") == r.get("eye", "") or "Unknown" in {ex.get("eye", ""), r.get("eye", "")}
            if not (same_source and same_name and phase_compatible and eye_compatible):
                continue

            best = ex if row_score(ex) >= row_score(r) else r
            if not best.get("frequency", ""):
                best["frequency"] = ex.get("frequency", "") or r.get("frequency", "")
            if not best.get("purpose", ""):
                best["purpose"] = ex.get("purpose", "") or r.get("purpose", "")
            if best.get("eye", "Unknown") == "Unknown":
                best["eye"] = ex.get("eye", "Unknown") if ex.get("eye", "Unknown") != "Unknown" else r.get("eye", "Unknown")
            if best.get("phase", "unknown") == "unknown":
                best["phase"] = ex.get("phase", "unknown") if ex.get("phase", "unknown") != "unknown" else r.get("phase", "unknown")
            out[idx] = best
            merged = True
            break
        if not merged:
            out.append(r)
    return out


def extract_stage_medications_from_text(
    text: str,
    source_record_id: str,
    surgery_eye: str,
    default_phase: str,
) -> List[Dict[str, Any]]:
    meds: List[Dict[str, Any]] = []
    for line in re.split(r"[\n\r]+", text):
        chunk = line.strip()
        if not chunk:
            continue
        has_med_hint = MED_NAME_RE.search(chunk) is not None or any(k in chunk for k in MED_KEYWORD_HINTS)
        if not has_med_hint and "目前用药" not in chunk:
            continue
        allow_generic = "目前用药" in chunk or "+" in chunk or "＋" in chunk
        phase = infer_med_phase(chunk, default_phase=default_phase)
        meds.extend(
            build_med_rows_from_block(
                text=chunk,
                source_record_id=source_record_id,
                phase=phase,
                surgery_eye=surgery_eye,
                allow_generic=allow_generic,
            )
        )
    return dedupe_med_rows(meds)


def extract_admission_medications(section: str, source_record_id: str, surgery_eye: str) -> List[Dict[str, Any]]:
    meds: List[Dict[str, Any]] = []

    for m in re.finditer(r"术前(?:予以|给予|用药)\s*([\s\S]{0,900}?)(?:治疗|根据病史|诊断明确|于[12][0-9]{3}|$)", section):
        meds.extend(
            build_med_rows_from_block(
                m.group(1),
                source_record_id=source_record_id,
                phase="pre_op",
                surgery_eye=surgery_eye,
            )
        )
    for m in re.finditer(r"术后(?:予以|给予|用药)\s*([\s\S]{0,600}?)(?:治疗|。|；|$)", section):
        meds.extend(
            build_med_rows_from_block(
                m.group(1),
                source_record_id=source_record_id,
                phase="post_op",
                surgery_eye=surgery_eye,
            )
        )
    for m in re.finditer(r"目前用药[:：]\s*([\s\S]{0,240}?)(?:。|；|,|，|专科检查[:：]|$)", section):
        meds.extend(
            build_med_rows_from_block(
                m.group(1),
                source_record_id=source_record_id,
                phase="chronic",
                surgery_eye=surgery_eye,
                allow_generic=True,
            )
        )

    return dedupe_med_rows(meds)


def extract_discharge_medications(section: str, source_record_id: str, surgery_eye: str) -> List[Dict[str, Any]]:
    meds: List[Dict[str, Any]] = []

    for m in re.finditer(r"术前(?:予以|给予|用药)\s*([\s\S]{0,900}?)(?:治疗|根据病史|诊断明确|于[12][0-9]{3}|$)", section):
        meds.extend(
            build_med_rows_from_block(
                m.group(1),
                source_record_id=source_record_id,
                phase="pre_op",
                surgery_eye=surgery_eye,
            )
        )

    for m in re.finditer(r"术后(?:予以|给予|用药)\s*([\s\S]{0,700}?)(?:治疗|。|；|$)", section):
        meds.extend(
            build_med_rows_from_block(
                m.group(1),
                source_record_id=source_record_id,
                phase="post_op",
                surgery_eye=surgery_eye,
            )
        )

    for m in re.finditer(r"目前用药[:：]\s*([\s\S]{0,120}?)(?:。|；|,|，|专科检查[:：]|$)", section):
        meds.extend(
            build_med_rows_from_block(
                m.group(1),
                source_record_id=source_record_id,
                phase="chronic",
                surgery_eye=surgery_eye,
                allow_generic=True,
            )
        )

    m_dis = re.search(r"出院带药[:：]\s*([\s\S]{0,2400}?)(?:出院医嘱[:：]|注意事项[:：]|医师签名[:：]|$)", section, re.S)
    if m_dis:
        meds.extend(
            build_med_rows_from_block(
                m_dis.group(1),
                source_record_id=source_record_id,
                phase="discharge",
                surgery_eye=surgery_eye,
                allow_generic=True,
            )
        )
        meds.extend(
            extract_stage_medications_from_text(
                text=m_dis.group(1),
                source_record_id=source_record_id,
                surgery_eye=surgery_eye,
                default_phase="discharge",
            )
        )

    return dedupe_med_rows(meds)


def extract_followup_medications(raw_text: str, source_record_id: str, surgery_eye: str) -> List[Dict[str, Any]]:
    meds: List[Dict[str, Any]] = []
    text = raw_text.replace("\u00a0", " ")

    for m in re.finditer(r"目前用药[:：]\s*([\s\S]{0,180}?)(?:。|；|,|，|专科检查[:：]|辅助检查[:：]|$)", text):
        meds.extend(
            build_med_rows_from_block(
                m.group(1),
                source_record_id=source_record_id,
                phase="followup",
                surgery_eye=surgery_eye,
                allow_generic=True,
            )
        )

    m_proc = re.search(r"处理[:：]\s*([\s\S]{0,800}?)(?:随访医嘱[:：]|辅助检查[:：]|初步诊断[:：]|$)", text, re.S)
    if m_proc:
        meds.extend(
            build_med_rows_from_block(
                m_proc.group(1),
                source_record_id=source_record_id,
                phase="followup",
                surgery_eye=surgery_eye,
            )
        )

    m_plan = re.search(r"随访医嘱[:：]\s*([\s\S]{0,700}?)(?:辅助检查[:：]|初步诊断[:：]|$)", text, re.S)
    if m_plan:
        meds.extend(
            build_med_rows_from_block(
                m_plan.group(1),
                source_record_id=source_record_id,
                phase="followup",
                surgery_eye=surgery_eye,
                allow_generic=True,
            )
        )

    meds.extend(
        extract_stage_medications_from_text(
            text=text,
            source_record_id=source_record_id,
            surgery_eye=surgery_eye,
            default_phase="followup",
        )
    )

    return dedupe_med_rows(meds)


def blank_eye_state() -> Dict[str, Any]:
    return {
        "iop": None,
        "ucva": "",
        "bcva": "",
        "axial_length": None,
        "central_corneal_thickness": None,
        "visual_field_md": None,
        "cup_to_disc_ratio": None,
        "rnfl_average_thickness": None,
        "anterior_chamber_depth": None,
        "angle_status": "",
        "slit_lamp_findings": "",
        "other_findings": "",
    }


def extract_preop_eye_facts(section_text: str) -> Dict[str, Dict[str, Any]]:
    out = {"OD": blank_eye_state(), "OS": blank_eye_state()}
    sec = section_text.replace("\u00a0", " ")

    # vision
    for eye, eye_cn in (("OD", "右眼"), ("OS", "左眼")):
        m_uc = re.search(rf"{eye_cn}裸眼[:：]\s*([^\s，,；;。]+)", sec)
        if m_uc:
            out[eye]["ucva"] = normalize_visual_token(m_uc.group(1))
        m_bc = re.search(
            rf"{eye_cn}裸眼[:：][\s\S]{{0,180}}?矫正视力[:：][\s\S]{{0,140}}?→\s*([^\s，,；;。]+)",
            sec,
            re.S,
        )
        if m_bc:
            out[eye]["bcva"] = normalize_visual_token(m_bc.group(1))

    # iop
    m_iop_line = re.search(r"眼压[:：]\s*([\s\S]{0,320}?)(?:。|入院诊断[:：]|右眼眼睑|左眼眼睑|诊疗经过[:：]|$)", sec, re.S)
    if m_iop_line:
        line = m_iop_line.group(1)
        m_od = re.search(r"(?:右眼|OD)(?:NCT)?\s*([0-9]+(?:\.[0-9]+)?)\s*mmHg", line, re.I)
        m_os = re.search(r"(?:左眼|OS)(?:NCT)?\s*([0-9]+(?:\.[0-9]+)?)\s*mmHg", line, re.I)
        if m_od:
            out["OD"]["iop"] = parse_float(m_od.group(1))
        if m_os:
            out["OS"]["iop"] = parse_float(m_os.group(1))

    # axial length
    m_ax = re.search(r"眼轴[\s\S]{0,300}?(?:。|；|入院诊断[:：]|$)", sec, re.S)
    if m_ax:
        line = m_ax.group(0)
        m_od = re.search(r"OD\s*([0-9]+(?:\.[0-9]+)?)\s*mm", line, re.I)
        m_os = re.search(r"OS\s*([0-9]+(?:\.[0-9]+)?)\s*mm", line, re.I)
        if m_od:
            out["OD"]["axial_length"] = parse_float(m_od.group(1))
        if m_os:
            out["OS"]["axial_length"] = parse_float(m_os.group(1))

    # visual field
    m_md_od = re.search(r"右眼MD\s*([\-0-9]+(?:\.[0-9]+)?)\s*dB", sec, re.I)
    m_md_os = re.search(r"左眼MD\s*([\-0-9]+(?:\.[0-9]+)?)\s*dB", sec, re.I)
    if not m_md_od:
        m_md_od = re.search(r"OD\s*MD\s*([\-0-9]+(?:\.[0-9]+)?)\s*dB", sec, re.I)
    if not m_md_os:
        m_md_os = re.search(r"OS\s*MD\s*([\-0-9]+(?:\.[0-9]+)?)\s*dB", sec, re.I)
    if m_md_od:
        out["OD"]["visual_field_md"] = parse_float(m_md_od.group(1))
    if m_md_os:
        out["OS"]["visual_field_md"] = parse_float(m_md_os.group(1))
    if not m_md_os and re.search(r"左眼[^。；\n]{0,120}(向心性缩小|视野缩小)", sec):
        out["OS"]["visual_field_md"] = "Concentric field constriction"

    # RNFL
    m_rnfl_os = re.search(r"OS[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:um|μm)", sec, re.I)
    if m_rnfl_os:
        out["OS"]["rnfl_average_thickness"] = parse_float(m_rnfl_os.group(1))

    # cup-disc ratio
    m_od_block = re.search(r"右眼([\s\S]{0,420}?)(?:左眼|4\.\s*辅助检查|入院诊断[:：]|诊疗经过[:：]|$)", sec, re.S)
    m_os_block = re.search(r"左眼([\s\S]{0,420}?)(?:4\.\s*辅助检查|入院诊断[:：]|诊疗经过[:：]|$)", sec, re.S)
    if m_od_block:
        m_cd = re.search(r"C/D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", m_od_block.group(1))
        if m_cd:
            out["OD"]["cup_to_disc_ratio"] = parse_float(m_cd.group(1))
    if m_os_block:
        m_cd = re.search(r"C/D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", m_os_block.group(1))
        if m_cd:
            out["OS"]["cup_to_disc_ratio"] = parse_float(m_cd.group(1))
    for eye, eye_token in (("OD", "OD"), ("OS", "OS")):
        if out[eye]["cup_to_disc_ratio"] is None:
            m_cd = re.search(rf"{eye_token}[\s\S]{{0,220}}?C/D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", sec, re.S)
            if m_cd:
                out[eye]["cup_to_disc_ratio"] = parse_float(m_cd.group(1))
    cds = re.findall(r"C/D\s*[=约]?\s*([0-9]+(?:\.[0-9]+)?)", sec)
    if out["OD"]["cup_to_disc_ratio"] is None and cds:
        out["OD"]["cup_to_disc_ratio"] = parse_float(cds[0])
    if out["OS"]["cup_to_disc_ratio"] is None:
        if len(cds) >= 2:
            out["OS"]["cup_to_disc_ratio"] = parse_float(cds[1])
        elif len(cds) == 1:
            out["OS"]["cup_to_disc_ratio"] = parse_float(cds[0])

    # angle status
    if "PAC约1/4CT" in sec or "房角狭窄" in sec:
        out["OD"]["angle_status"] = "Narrow"
        out["OS"]["angle_status"] = "Narrow"
    if "房角开放" in sec or ">1/2CT" in sec:
        out["OD"]["angle_status"] = "Open"
        out["OS"]["angle_status"] = "Open"
    if "小梁切除术后" in sec:
        out["OS"]["angle_status"] = out["OS"]["angle_status"] or "Post-trab status"

    return out


def extract_postop_eye_facts(section_text: str, surgery_eye: str) -> Dict[str, Dict[str, Any]]:
    out = {"OD": blank_eye_state(), "OS": blank_eye_state()}
    sec = section_text.replace("\u00a0", " ")
    m_post = re.search(r"出院情况[:：]\s*(.*?)(?:出院医嘱[:：]|出院带药[:：]|医师签名[:：]|$)", sec, re.S)
    chunk = m_post.group(1) if m_post else sec

    for eye, eye_cn in (("OD", "右眼"), ("OS", "左眼")):
        m_va = re.search(rf"{eye_cn}远视力[:：]\s*([^\s，,；;。]+)", chunk)
        if m_va:
            out[eye]["ucva"] = normalize_visual_token(m_va.group(1))
        m_iop = re.search(rf"{eye_cn}眼压[:：]?(?:NCT)?\s*([0-9]+(?:\.[0-9]+)?)\s*mmHg", chunk, re.I)
        if m_iop:
            out[eye]["iop"] = parse_float(m_iop.group(1))

    # concise slit-lamp summary for operated eye if present
    m_op = re.search(r"术眼([^。；\n]{0,220})", chunk)
    if m_op and surgery_eye in {"OD", "OS"}:
        out[surgery_eye]["slit_lamp_findings"] = re.sub(r"\s+", " ", m_op.group(1)).strip("，,。；; ")
    if "对侧眼查体同入院" in chunk and surgery_eye in {"OD", "OS"}:
        fellow = "OS" if surgery_eye == "OD" else "OD"
        out[fellow]["slit_lamp_findings"] = "对侧眼情况同入院"

    return out


def extract_surgery_events_from_text(text: str, fallback_eye: str = "Unknown") -> List[Dict[str, str]]:
    events: List[Dict[str, str]] = []
    sec = text.replace("\u00a0", " ")

    # Primary surgery event with date preceding the quoted surgery name.
    for m in re.finditer(
        r"于\s*([12][0-9]{3}[年./-][0-9]{1,2}[月./-][0-9]{1,2})[\s\S]{0,420}?行[\s\S]{0,80}?[“\"]([\s\S]{4,320}?)[”\"]",
        sec,
    ):
        date = normalize_date(m.group(1))
        raw_name = re.sub(r"\s+", " ", m.group(2)).strip()
        if "术" not in raw_name and "PEI" not in raw_name and "Trab" not in raw_name:
            continue
        nearby = sec[max(0, m.start() - 40) : min(len(sec), m.end() + 40)]
        eye = infer_eye_from_text(nearby, fallback_eye)
        events.append(
            {
                "date": date,
                "eye": eye,
                "name": normalize_surgery_name(raw_name, eye),
            }
        )

    # Additional procedure events (laser/suture lysis).
    for m in re.finditer(
        r"([12][0-9]{3}[./-][0-9]{1,2}[./-][0-9]{1,2})[^。；\n]{0,80}(激光[^。；\n]{0,60}|拆[线除][^。；\n]{0,60})",
        sec,
    ):
        date = normalize_date(m.group(1))
        phrase = re.sub(r"\s+", " ", m.group(2)).strip()
        nearby = sec[max(0, m.start() - 40) : min(len(sec), m.end() + 40)]
        eye = infer_eye_from_text(nearby, fallback_eye)
        events.append(
            {
                "date": date,
                "eye": eye,
                "name": normalize_surgery_name(phrase, eye),
            }
        )

    dedup: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for e in events:
        if not e.get("date"):
            continue
        key = (e["date"], e["eye"], e["name"])
        if key not in dedup:
            dedup[key] = e
    out = list(dedup.values())
    out.sort(key=lambda x: x["date"])
    return out


def extract_discharge_episode_facts(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    header_re = re.compile(
        r"入院日期[:：]\s*([12][0-9]{3}[./-][0-9]{1,2}[./-][0-9]{1,2})[\s\S]{0,100}?出院日期[:：]\s*([12][0-9]{3}[./-][0-9]{1,2}[./-][0-9]{1,2})",
        re.S,
    )
    for r in records:
        if r.get("record_type") != "discharge":
            continue
        raw = r.get("_raw_text", "") or r.get("text", "")
        text = raw.replace("\u00a0", " ").replace("\r\n", "\n")
        sections: List[Dict[str, str]] = []
        matches = list(header_re.finditer(text))
        if matches:
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append(
                    {
                        "section": text[start:end],
                        "admission_date": normalize_date(m.group(1)),
                        "discharge_date": normalize_date(m.group(2)),
                    }
                )
        else:
            m_adm = re.search(r"入院日期[:：]\s*([12][0-9]{3}[./-][0-9]{1,2}[./-][0-9]{1,2})", text)
            m_dis = re.search(r"出院日期[:：]\s*([12][0-9]{3}[./-][0-9]{1,2}[./-][0-9]{1,2})", text)
            sections.append(
                {
                    "section": text,
                    "admission_date": normalize_date(m_adm.group(1)) if m_adm else "",
                    "discharge_date": normalize_date(m_dis.group(1)) if m_dis else "",
                }
            )

        for sec_meta in sections:
            section = sec_meta["section"]
            adm = sec_meta["admission_date"]
            dis = sec_meta["discharge_date"]

            m_course = re.search(r"诊疗经过[:：]\s*(.*?)(?:出院诊断[:：]|出院情况[:：]|$)", section, re.S)
            surgery_source = m_course.group(1) if m_course else section
            surgery_events = extract_surgery_events_from_text(surgery_source)
            if not surgery_events:
                surgery_events = extract_surgery_events_from_text(section)
            surgery_eye = surgery_events[0]["eye"] if surgery_events else "Unknown"

            pre = extract_preop_eye_facts(section)
            post = extract_postop_eye_facts(section, surgery_eye=surgery_eye)
            meds = extract_discharge_medications(
                section=section,
                source_record_id=r["record_id"],
                surgery_eye=surgery_eye,
            )
            current_meds = sorted({m["name"] for m in meds if m.get("phase") in {"pre_op", "chronic", "discharge", "followup"}})

            m_doc = re.search(r"医师签名[:：]\s*(DR_[0-9]{6})", section)
            doctor_id = m_doc.group(1) if m_doc else "DR_000000"

            m_diag = re.search(r"入院诊断[:：]\s*(.*?)(?:诊疗经过[:：]|出院诊断[:：])", section, re.S)
            diag_text = ""
            diag_list: List[str] = []
            if m_diag:
                diag_text = re.sub(r"\s+", " ", m_diag.group(1)).strip(" ;；。")
                if diag_text:
                    for part in re.split(r"[;；]+", diag_text):
                        part = re.sub(r"^\s*[0-9]+\s*[.、]?\s*", "", part.strip())
                        if part:
                            diag_list.append(part)

            duration = None
            m_duration = re.search(r"病程\s*([0-9]+)\s*年", section)
            if m_duration:
                duration = int(m_duration.group(1))

            # Normalize some common surgery aliases when no primary surgery found.
            if not surgery_events:
                for m_alt in re.finditer(
                    r"(左眼|右眼)[:：]?\s*([A-Za-z+]{2,40})\s*([12][0-9]{3}[./-][0-9]{1,2}[./-][0-9]{1,2})",
                    section,
                ):
                    eye = "OS" if m_alt.group(1) == "左眼" else "OD"
                    name = normalize_surgery_name(m_alt.group(2), eye)
                    surgery_events.append(
                        {
                            "date": normalize_date(m_alt.group(3)),
                            "eye": eye,
                            "name": name,
                        }
                    )
                surgery_events.sort(key=lambda x: x["date"])

            facts.append(
                {
                    "admission_date": adm,
                    "discharge_date": dis,
                    "source_record_id": r["record_id"],
                    "doctor_id": doctor_id,
                    "surgeries": surgery_events,
                    "diagnoses_raw": diag_text,
                    "diagnoses_list": diag_list,
                    "duration_years": duration,
                    "s1_eyes_data": pre,
                    "s2_eyes_data": post,
                    "medications": meds,
                    "current_medications": current_meds,
                }
            )

    facts.sort(key=lambda x: (x.get("admission_date", ""), x.get("discharge_date", "")))
    return facts


def extract_surgery_anchors(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    for record in records:
        text = record.get("_raw_text", "") or record.get("text", "")
        # Prefer surgery events with explicit "于YYYY-MM-DD ... “手术名”" pattern.
        for ev in extract_surgery_events_from_text(text):
            if "Laser suture lysis" in ev["name"]:
                continue
            anchors.append(
                {
                    "date": ev["date"],
                    "eye": ev["eye"],
                    "name": ev["name"],
                    "source_record_id": record["record_id"],
                }
            )
        # quoted fragments: robust unicode quote matching (allow wrapped lines)
        for m in re.finditer(r"[\u201c\"]([\s\S]{4,260}?)[\u201d\"]", text):
            raw_name = re.sub(r"\s+", " ", m.group(1)).strip()
            name = normalize_surgery_name(raw_name, "Unknown")
            if "青光眼" in name and "术后" in name:
                continue
            if "术" not in name and "PEI" not in name and "Trab" not in name:
                continue
            if "Laser suture lysis" in name:
                continue
            left_start = max(0, m.start() - 2200)
            right_end = min(len(text), m.end() + 220)
            neighborhood = text[left_start:right_end]
            date_hits: List[Tuple[int, str]] = []
            prev_date_hits: List[Tuple[int, str]] = []
            for dm in re.finditer(r"([12][0-9]{3}[年./-][0-9]{1,2}[月./-][0-9]{1,2})", neighborhood):
                global_pos = left_start + dm.start()
                date_hits.append((abs(global_pos - m.start()), dm.group(1)))
                if global_pos <= m.start():
                    prev_date_hits.append((global_pos, dm.group(1)))
            date = ""
            if prev_date_hits:
                prev_date_hits.sort(key=lambda x: x[0], reverse=True)
                date = normalize_date(prev_date_hits[0][1])
            elif date_hits:
                date_hits.sort(key=lambda x: x[0])
                date = normalize_date(date_hits[0][1])
            if not date:
                date = record.get("document_date", "")

            eye = "Unknown"
            if "左眼" in name:
                eye = "OS"
            elif "右眼" in name:
                eye = "OD"
            elif "双眼" in name:
                eye = "OU"
            anchors.append(
                {
                    "date": date,
                    "eye": eye,
                    "name": name,
                    "source_record_id": record["record_id"],
                }
            )
        # compact style: 左眼：PEI+Trab2024.10.10
        for m in re.finditer(
            r"(左眼|右眼)[:：]?\s*([A-Za-z+]{2,40})\s*([12][0-9]{3}[./-][0-9]{1,2}[./-][0-9]{1,2})",
            text,
        ):
            eye = "OS" if m.group(1) == "左眼" else "OD"
            name = normalize_surgery_name(m.group(2).strip(), eye)
            date = normalize_date(m.group(3))
            anchors.append(
                {
                    "date": date,
                    "eye": eye,
                    "name": name,
                    "source_record_id": record["record_id"],
                }
            )
    dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for a in anchors:
        if not a["date"]:
            continue
        key = (a["date"], a["eye"], a["name"])
        if key not in dedup:
            dedup[key] = a
    out = list(dedup.values())
    out.sort(key=lambda x: x["date"])
    return out


def build_prompt(template: str, schema_json: Dict[str, Any], hint_json: Dict[str, Any], patient_id: str, glaucoma_type: str, records: List[Dict[str, Any]]) -> str:
    compact_schema, compact_hint = build_compact_prompt_schema(schema_json, hint_json)
    context = {
        "patient_id": patient_id,
        "glaucoma_type": glaucoma_type,
        "source_records": [
            {k: v for k, v in r.items() if k != "text"} for r in records
        ],
    }
    blocks = []
    for r in records:
        blocks.append(
            "\n".join(
                [
                    f"### RECORD {r['record_id']} (type={r['record_type']}, index={r['record_index']}, date={r['document_date']})",
                    f"source_path={r['source_path']}",
                    f"hash_sha256={r['hash_sha256']}",
                    r["text"],
                ]
            )
        )
    text_block = "\n\n".join(blocks)

    out = template
    out = out.replace("<<SCHEMA_JSON>>", json.dumps(compact_schema, ensure_ascii=False, indent=2))
    out = out.replace("<<SCHEMA_HINT_JSON>>", json.dumps(compact_hint, ensure_ascii=False, indent=2))
    out = out.replace("<<PATIENT_CONTEXT_JSON>>", json.dumps(context, ensure_ascii=False, indent=2))
    out = out.replace("<<EHR_TEXT_BLOCK>>", text_block)
    return out


def parse_json_from_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    for c in m:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    starts = [x.start() for x in re.finditer(r"\{", text)]
    for start in starts:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start : i + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
    raise ValueError("Cannot parse JSON object from model response")


def merge_template(template: Any, payload: Any) -> Any:
    if payload is None:
        return copy.deepcopy(template)
    if isinstance(template, dict):
        src = payload if isinstance(payload, dict) else {}
        out: Dict[str, Any] = {}
        for k, tv in template.items():
            out[k] = merge_template(tv, src.get(k))
        return out
    if isinstance(template, list):
        if not isinstance(payload, list):
            return []
        if not template:
            return copy.deepcopy(payload)
        return [merge_template(template[0], x) for x in payload]
    return payload


def ensure_id(value: str, kind: str, fallback: str) -> str:
    if isinstance(value, str) and ID_PATTERNS[kind].match(value):
        return value
    return fallback


def build_final(schema: Dict[str, Any], payload: Dict[str, Any], patient_id: str, glaucoma_type: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnum = int(patient_id.split("_")[1])
    now_iso = datetime.now().astimezone().isoformat(timespec="seconds")

    root = {k: copy.deepcopy(v) for k, v in schema.items()}
    root["hospitals"] = []
    root["doctors"] = []
    root["patients"] = []
    root["evidence_catalog"] = []

    h_tpl = schema["hospitals"][0]
    d_tpl = schema["doctors"][0]
    p_tpl = schema["patients"][0]
    ep_tpl = p_tpl["episodes"][0]
    fu_tpl = ep_tpl["follow_up_state"]["followup_records"][0]
    ev_tpl = schema["evidence_catalog"][0]

    hs = payload.get("hospitals") or [{"hospital_id": "HP_000000", "hospital_name": ""}]
    hs = [merge_template(h_tpl, x) for x in hs]
    for h in hs:
        h["hospital_id"] = ensure_id(h.get("hospital_id", ""), "hospital", "HP_000000")
    root["hospitals"] = hs

    docs = [merge_template(d_tpl, x) for x in (payload.get("doctors") or [])]
    for i, d in enumerate(docs):
        d["doctor_id"] = ensure_id(d.get("doctor_id", ""), "doctor", f"DR_{i:06d}")
    root["doctors"] = docs

    pat = merge_template(p_tpl, {})
    pat["patient_id"] = patient_id
    pat["hospital_id"] = hs[0]["hospital_id"]
    pat["glaucoma_type"] = glaucoma_type
    pat["text_source_paths"] = [r["source_path"] for r in records]
    pat["source_records"] = [
        {
            k: v
            for k, v in r.items()
            if k not in {"text", "_raw_text", "_clean_text"}
        }
        for r in records
    ]
    pat["patient_demographics"] = merge_template(p_tpl["patient_demographics"], payload.get("patient_demographics", {}))
    pat["patient_status_tracking"] = merge_template(p_tpl["patient_status_tracking"], payload.get("patient_status_tracking", {}))
    if not pat["patient_status_tracking"].get("extraction_status"):
        pat["patient_status_tracking"]["extraction_status"] = "complete"
    if not pat["patient_status_tracking"].get("verification_status"):
        pat["patient_status_tracking"]["verification_status"] = "machine_checked"
    if not pat["patient_status_tracking"].get("last_updated_at"):
        pat["patient_status_tracking"]["last_updated_at"] = now_iso

    record_ids = {r["record_id"] for r in records}
    episodes = []
    for ep_idx, raw_ep in enumerate(payload.get("episodes") or []):
        ep = merge_template(ep_tpl, raw_ep)
        ep["episode_id"] = ensure_id(ep.get("episode_id", ""), "episode", f"EP_{pnum:03d}{ep_idx:03d}")
        ep["episode_index"] = ep_idx
        ep["linked_record_ids"] = [x for x in ep.get("linked_record_ids", []) if x in record_ids]
        s1 = ep.get("s1_pre_intervention_state", {})
        if isinstance(s1, dict):
            gd = s1.get("glaucoma_diagnosis_type", "")
            if gd not in {"AACG", "CACG", "POAG", "Unknown"}:
                s1["glaucoma_diagnosis_type"] = (
                    glaucoma_type if glaucoma_type in {"AACG", "CACG", "POAG"} else "Unknown"
                )
            ep["s1_pre_intervention_state"] = s1
        fus = []
        for fu_idx, raw_fu in enumerate(ep.get("follow_up_state", {}).get("followup_records", [])):
            fu = merge_template(fu_tpl, raw_fu)
            fu["followup_id"] = ensure_id(fu.get("followup_id", ""), "followup", f"FU_{pnum:03d}{ep_idx}{fu_idx:02d}")
            fu["followup_index"] = fu_idx
            if fu.get("source_record_id", "") not in record_ids:
                fu["source_record_id"] = ""
            fus.append(fu)
        ep["follow_up_state"]["followup_records"] = fus

        st = ep.get("episode_status_tracking", {})
        if not st.get("episode_status"):
            st["episode_status"] = "active" if fus else "closed"
        if not st.get("follow_up_state_status"):
            st["follow_up_state_status"] = "complete" if fus else "missing"
        if st.get("data_completeness_score", None) is None:
            st["data_completeness_score"] = 0.75 if fus else 0.5
        if not st.get("verification_status"):
            st["verification_status"] = "machine_checked"
        if not st.get("verified_by"):
            st["verified_by"] = "llm-pipeline"
        if not st.get("verified_at"):
            st["verified_at"] = now_iso
        ep["episode_status_tracking"] = st
        episodes.append(ep)

    pat["episodes"] = episodes
    root["patients"] = [pat]

    evs = [merge_template(ev_tpl, x) for x in (payload.get("evidence_catalog") or [])]
    ep_ids = {x["episode_id"] for x in episodes}
    fu_ids = {fu["followup_id"] for x in episodes for fu in x["follow_up_state"]["followup_records"]}
    for i, ev in enumerate(evs):
        ev["evidence_id"] = ensure_id(ev.get("evidence_id", ""), "evidence", f"EV_{pnum:03d}{i:03d}")
        ev["patient_id"] = ensure_id(ev.get("patient_id", ""), "patient", patient_id)
        if ev.get("episode_id", "") not in ep_ids:
            ev["episode_id"] = ""
        if ev.get("followup_id", "") not in fu_ids:
            ev["followup_id"] = ""
        if ev.get("record_id", "") not in record_ids:
            ev["record_id"] = ""
        if ev.get("verifiable", "") not in {"yes", "no"}:
            ev["verifiable"] = "no"
    root["evidence_catalog"] = evs

    ev_id_set = {x["evidence_id"] for x in evs}
    for ep in episodes:
        ep["evidence_refs"] = [x for x in ep.get("evidence_refs", []) if x in ev_id_set]
        for sk in ["s1_pre_intervention_state", "a_treatment_action", "s2_post_surgery_state"]:
            ep[sk]["evidence_refs"] = [x for x in ep[sk].get("evidence_refs", []) if x in ev_id_set]
        for fu in ep["follow_up_state"]["followup_records"]:
            fu["evidence_refs"] = [x for x in fu.get("evidence_refs", []) if x in ev_id_set]

    return root


def validate_refs(final_json: Dict[str, Any]) -> List[str]:
    warns: List[str] = []
    patients = final_json.get("patients", [])
    if len(patients) != 1:
        warns.append(f"Expected 1 patient, got {len(patients)}")
        return warns
    p = patients[0]
    rec_ids = {x["record_id"] for x in p.get("source_records", [])}
    ev_ids = {x["evidence_id"] for x in final_json.get("evidence_catalog", [])}
    for ep in p.get("episodes", []):
        for rid in ep.get("linked_record_ids", []):
            if rid not in rec_ids:
                warns.append(f"Unknown linked_record_id: {rid}")
        for sk in ["s1_pre_intervention_state", "a_treatment_action", "s2_post_surgery_state"]:
            for ev in ep.get(sk, {}).get("evidence_refs", []):
                if ev not in ev_ids:
                    warns.append(f"Unknown evidence_ref in {sk}: {ev}")
        for fu in ep.get("follow_up_state", {}).get("followup_records", []):
            rid = fu.get("source_record_id", "")
            if rid and rid not in rec_ids:
                warns.append(f"Unknown followup source_record_id: {rid}")
    return warns


def fill_patient_demographics_from_records(final_json: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    patients = final_json.get("patients", [])
    if len(patients) != 1:
        return final_json
    patient = patients[0]
    demo = patient.get("patient_demographics", {})
    if not isinstance(demo, dict):
        demo = {}

    text_blocks: List[str] = []
    for r in records:
        txt = r.get("_raw_text", "") or r.get("text", "")
        if isinstance(txt, str) and txt:
            text_blocks.append(txt)
    text = "\n".join(text_blocks).replace("\u00a0", " ")

    # age_first_seen
    if demo.get("age_first_seen", None) is None:
        m_age = re.search(r"年龄[:：]?\s*([0-9]{1,3})\s*岁", text)
        if m_age:
            try:
                demo["age_first_seen"] = int(m_age.group(1))
            except Exception:  # pylint: disable=broad-except
                pass

    # biological_sex
    sex = str(demo.get("biological_sex", "") or "").strip()
    if sex not in {"M", "F", "Unknown"}:
        m_sex = re.search(r"性别[:：]?\s*(男|女|M|F)", text, re.I)
        if m_sex:
            tok = m_sex.group(1).upper()
            if tok in {"男", "M"}:
                demo["biological_sex"] = "M"
            elif tok in {"女", "F"}:
                demo["biological_sex"] = "F"
        if str(demo.get("biological_sex", "")).strip() not in {"M", "F"}:
            demo["biological_sex"] = "Unknown"

    race = str(demo.get("race", "") or "").strip()
    ethnicity = str(demo.get("ethnicity", "") or "").strip()

    # Required mapping by project rule:
    # if Han ethnicity is detected, map to Asian / Chinese.
    han_detected = False
    if "汉族" in text:
        han_detected = True
    else:
        m_nation = re.search(r"民族[:：]?\s*([^\s，,；;。|]+)", text)
        if m_nation and m_nation.group(1) == "汉族":
            han_detected = True

    if han_detected:
        if not race:
            demo["race"] = "Asian"
        if not ethnicity:
            demo["ethnicity"] = "Chinese"
    else:
        if not race and re.search(r"\bAsian\b|亚洲", text, re.I):
            demo["race"] = "Asian"
        if not ethnicity and re.search(r"\bChinese\b|华人|中国", text, re.I):
            demo["ethnicity"] = "Chinese"

    patient["patient_demographics"] = demo
    final_json["patients"][0] = patient
    return final_json


def auto_build_evidence(final_json: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    patients = final_json.get("patients", [])
    if len(patients) != 1:
        return final_json
    p = patients[0]
    pid = p.get("patient_id", "")
    if not ID_PATTERNS["patient"].match(pid):
        return final_json
    pnum = int(pid.split("_")[1])

    record_map: Dict[str, Dict[str, Any]] = {r["record_id"]: r for r in records}
    source_records = {x["record_id"]: x for x in p.get("source_records", []) if isinstance(x, dict) and x.get("record_id")}
    for rid, sr in source_records.items():
        if rid not in record_map:
            record_map[rid] = sr

    evidence_catalog: List[Dict[str, Any]] = []
    counter = 0

    def pick_record(ep: Dict[str, Any], preferred_types: List[str]) -> str:
        linked = ep.get("linked_record_ids", [])
        for t in preferred_types:
            for rid in linked:
                if record_map.get(rid, {}).get("record_type") == t:
                    return rid
        return linked[0] if linked else ""

    def build_snippet(record_id: str, keyword: str) -> str:
        rec = record_map.get(record_id, {})
        txt = rec.get("_clean_text", "") or rec.get("text", "")
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            return ""
        if keyword:
            idx = txt.find(keyword)
            if idx >= 0:
                st = max(0, idx - 48)
                ed = min(len(txt), idx + 128)
                return txt[st:ed]
        return txt[:176]

    def add_ev(
        field_path: str,
        extracted_value: Any,
        episode_id: str,
        record_id: str = "",
        followup_id: str = "",
        keyword: str = "",
    ) -> str:
        nonlocal counter
        ev_id = f"EV_{pnum:03d}{counter:03d}"
        counter += 1
        rec = record_map.get(record_id, {})
        src_path = rec.get("source_path", "")
        snippet = build_snippet(record_id, keyword)
        evidence_catalog.append(
            {
                "evidence_id": ev_id,
                "patient_id": pid,
                "episode_id": episode_id,
                "followup_id": followup_id,
                "record_id": record_id,
                "field_path": field_path,
                "extracted_value": str(extracted_value) if extracted_value is not None else "",
                "snippet": snippet,
                "source_path": src_path,
                "confidence": 0.95,
                "extraction_method": "record_level_parser+llm_merge",
                "annotator": "auto-evidence",
                "verifiable": "yes" if record_id and snippet else "no",
            }
        )
        return ev_id

    def non_empty(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str) and v == "":
            return False
        if isinstance(v, list) and not v:
            return False
        return True

    for ep_idx, ep in enumerate(p.get("episodes", [])):
        ep_id = ep.get("episode_id", f"EP_{pnum:03d}{ep_idx:03d}")
        ep["evidence_refs"] = []
        s1_refs: List[str] = []
        a_refs: List[str] = []
        s2_refs: List[str] = []

        adm_rid = pick_record(ep, ["admission", "discharge"])
        dis_rid = pick_record(ep, ["discharge", "admission"])

        adm_date = ep.get("hospitalization_info", {}).get("admission_date", "")
        if non_empty(adm_date):
            ev = add_ev(
                field_path=f"patients[0].episodes[{ep_idx}].hospitalization_info.admission_date",
                extracted_value=adm_date,
                episode_id=ep_id,
                record_id=adm_rid,
                keyword="入院日期",
            )
            ep["evidence_refs"].append(ev)

        dis_date = ep.get("hospitalization_info", {}).get("discharge_date", "")
        if non_empty(dis_date):
            ev = add_ev(
                field_path=f"patients[0].episodes[{ep_idx}].hospitalization_info.discharge_date",
                extracted_value=dis_date,
                episode_id=ep_id,
                record_id=dis_rid,
                keyword="出院日期",
            )
            ep["evidence_refs"].append(ev)

        for s_idx, surg in enumerate(ep.get("a_treatment_action", {}).get("surgeries", [])):
            rid = surg.get("source_record_id", "") or dis_rid
            val = json.dumps(
                {
                    "date": surg.get("date", ""),
                    "eye": surg.get("eye", ""),
                    "name": surg.get("name", ""),
                },
                ensure_ascii=False,
            )
            ev = add_ev(
                field_path=f"patients[0].episodes[{ep_idx}].a_treatment_action.surgeries[{s_idx}]",
                extracted_value=val,
                episode_id=ep_id,
                record_id=rid,
                keyword=surg.get("name", ""),
            )
            a_refs.append(ev)
            ep["evidence_refs"].append(ev)

        for m_idx, med in enumerate(ep.get("a_treatment_action", {}).get("medications", [])):
            if not non_empty(med.get("name", "")):
                continue
            rid = med.get("source_record_id", "") or dis_rid or adm_rid
            val = json.dumps(
                {
                    "phase": med.get("phase", ""),
                    "eye": med.get("eye", ""),
                    "name": med.get("name", ""),
                    "frequency": med.get("frequency", ""),
                    "purpose": med.get("purpose", ""),
                },
                ensure_ascii=False,
            )
            ev = add_ev(
                field_path=f"patients[0].episodes[{ep_idx}].a_treatment_action.medications[{m_idx}]",
                extracted_value=val,
                episode_id=ep_id,
                record_id=rid,
                keyword=med.get("name", ""),
            )
            a_refs.append(ev)
            ep["evidence_refs"].append(ev)

        s1 = ep.get("s1_pre_intervention_state", {})
        for eye in ("OD", "OS"):
            eye_data = s1.get("eyes_data", {}).get(eye, {})
            compact = {k: v for k, v in eye_data.items() if non_empty(v)}
            if compact:
                ev = add_ev(
                    field_path=f"patients[0].episodes[{ep_idx}].s1_pre_intervention_state.eyes_data.{eye}",
                    extracted_value=json.dumps(compact, ensure_ascii=False),
                    episode_id=ep_id,
                    record_id=adm_rid,
                    keyword="专科检查",
                )
                s1_refs.append(ev)
                ep["evidence_refs"].append(ev)

        s2 = ep.get("s2_post_surgery_state", {})
        for eye in ("OD", "OS"):
            eye_data = s2.get("eyes_data", {}).get(eye, {})
            compact = {k: v for k, v in eye_data.items() if non_empty(v)}
            if compact:
                ev = add_ev(
                    field_path=f"patients[0].episodes[{ep_idx}].s2_post_surgery_state.eyes_data.{eye}",
                    extracted_value=json.dumps(compact, ensure_ascii=False),
                    episode_id=ep_id,
                    record_id=dis_rid,
                    keyword="出院情况",
                )
                s2_refs.append(ev)
                ep["evidence_refs"].append(ev)

        ep["s1_pre_intervention_state"]["evidence_refs"] = sorted(set(s1_refs))
        ep["a_treatment_action"]["evidence_refs"] = sorted(set(a_refs))
        ep["s2_post_surgery_state"]["evidence_refs"] = sorted(set(s2_refs))
        ep["evidence_refs"] = sorted(set(ep["evidence_refs"]))

        for fu_idx, fu in enumerate(ep.get("follow_up_state", {}).get("followup_records", [])):
            rid = fu.get("source_record_id", "")
            fu_refs: List[str] = []
            if non_empty(fu.get("followup_date", "")):
                ev = add_ev(
                    field_path=f"patients[0].episodes[{ep_idx}].follow_up_state.followup_records[{fu_idx}].followup_date",
                    extracted_value=fu.get("followup_date", ""),
                    episode_id=ep_id,
                    followup_id=fu.get("followup_id", ""),
                    record_id=rid,
                    keyword="就诊日期",
                )
                fu_refs.append(ev)
                ep["evidence_refs"].append(ev)
            for eye in ("OD", "OS"):
                eye_data = fu.get("eyes_data", {}).get(eye, {})
                compact = {k: v for k, v in eye_data.items() if non_empty(v)}
                if compact:
                    ev = add_ev(
                        field_path=f"patients[0].episodes[{ep_idx}].follow_up_state.followup_records[{fu_idx}].eyes_data.{eye}",
                        extracted_value=json.dumps(compact, ensure_ascii=False),
                        episode_id=ep_id,
                        followup_id=fu.get("followup_id", ""),
                        record_id=rid,
                        keyword="专科检查",
                    )
                    fu_refs.append(ev)
                    ep["evidence_refs"].append(ev)
            fu["evidence_refs"] = sorted(set(fu_refs))

        ep["evidence_refs"] = sorted(set(ep["evidence_refs"]))

    final_json["evidence_catalog"] = evidence_catalog
    return final_json


def apply_record_level_refinement(
    final_json: Dict[str, Any],
    schema: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    patients = final_json.get("patients", [])
    if len(patients) != 1:
        return final_json
    patient = patients[0]
    patient_id = patient.get("patient_id", "")
    if not ID_PATTERNS["patient"].match(patient_id):
        return final_json
    pnum = int(patient_id.split("_")[1])
    def infer_laterality(text: str) -> str:
        if "双眼" in text or "OU" in text:
            return "OU"
        if "右眼" in text or "OD" in text:
            return "OD"
        if "左眼" in text or "OS" in text:
            return "OS"
        return "Unknown"

    def infer_stage(text: str) -> str:
        if "晚期" in text or "advanced" in text.lower():
            return "Advanced"
        if "终末期" in text or "end-stage" in text.lower():
            return "End-stage"
        if "中期" in text or "moderate" in text.lower():
            return "Moderate"
        if "早期" in text or "early" in text.lower():
            return "Early"
        return "Not Assessed"

    def overlay_eye(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if v is None:
                continue
            if isinstance(v, str) and v == "":
                continue
            dst[k] = v

    def has_eye_content(eye_data: Dict[str, Any]) -> bool:
        for v in eye_data.values():
            if v is None:
                continue
            if isinstance(v, str) and v == "":
                continue
            return True
        return False

    ep_tpl = schema["patients"][0]["episodes"][0]
    fu_tpl = ep_tpl["follow_up_state"]["followup_records"][0]
    med_tpl = ep_tpl["a_treatment_action"]["medications"][0]

    existing_fu_by_record: Dict[str, Dict[str, Any]] = {}
    for ep in patient.get("episodes", []):
        for fu in ep.get("follow_up_state", {}).get("followup_records", []):
            rid = fu.get("source_record_id", "")
            if rid:
                existing_fu_by_record[rid] = fu

    discharge_facts = extract_discharge_episode_facts(records)
    anchors = extract_surgery_anchors(records)

    admission_record_ids = [r["record_id"] for r in records if r.get("record_type") == "admission"]
    discharge_record_ids = [r["record_id"] for r in records if r.get("record_type") == "discharge"]

    episode_seeds: List[Dict[str, Any]] = []
    if discharge_facts:
        for idx, fact in enumerate(discharge_facts):
            surgeries = copy.deepcopy(fact.get("surgeries", []))
            if not surgeries and anchors:
                fallback = anchors[idx] if idx < len(anchors) else anchors[-1]
                surgeries = [
                    {
                        "date": fallback.get("date", ""),
                        "eye": fallback.get("eye", "Unknown"),
                        "name": normalize_surgery_name(fallback.get("name", "Surgery"), fallback.get("eye", "Unknown")),
                    }
                ]
            if not surgeries:
                base_date = fact.get("discharge_date", "") or fact.get("admission_date", "")
                surgeries = [
                    {
                        "date": base_date,
                        "eye": "Unknown",
                        "name": "Surgery",
                    }
                ]

            diag_list = fact.get("diagnoses_list", [])
            primary_condition = diag_list[0] if diag_list else ""
            diagnosis_text = " ".join(diag_list) if diag_list else ""
            episode_seeds.append(
                {
                    "admission_date": fact.get("admission_date", ""),
                    "discharge_date": fact.get("discharge_date", ""),
                    "source_record_id": fact.get("source_record_id", ""),
                    "doctor_id": fact.get("doctor_id", "DR_000000"),
                    "surgeries": surgeries,
                    "medications": fact.get("medications", []),
                    "s1_eyes_data": fact.get("s1_eyes_data", {"OD": {}, "OS": {}}),
                    "s2_eyes_data": fact.get("s2_eyes_data", {"OD": {}, "OS": {}}),
                    "ophthalmic_diagnoses": {
                        "primary_condition": primary_condition,
                        "primary_condition_laterality": infer_laterality(diagnosis_text),
                        "primary_condition_stage": infer_stage(diagnosis_text),
                        "primary_condition_duration": fact.get("duration_years", None),
                        "secondary_conditions": diag_list[1:] if len(diag_list) > 1 else [],
                        "current_medications": fact.get("current_medications", []),
                    },
                }
            )
    else:
        if not anchors:
            return final_json
        all_adm_dates: List[str] = []
        all_dis_dates: List[str] = []
        for r in records:
            raw = r.get("_raw_text", "")
            for m in re.finditer(r"入院(?:日期|时间)[:：]\s*([0-9]{4}[-./][0-9]{1,2}[-./][0-9]{1,2})", raw):
                d = normalize_date(m.group(1))
                if d:
                    all_adm_dates.append(d)
            for m in re.finditer(r"出院日期[:：]\s*([0-9]{4}[-./][0-9]{1,2}[-./][0-9]{1,2})", raw):
                d = normalize_date(m.group(1))
                if d:
                    all_dis_dates.append(d)
        all_adm_dates = sorted(set(all_adm_dates))
        all_dis_dates = sorted(set(all_dis_dates))
        for idx, anchor in enumerate(anchors):
            episode_seeds.append(
                {
                    "admission_date": all_adm_dates[idx] if idx < len(all_adm_dates) else "",
                    "discharge_date": all_dis_dates[idx] if idx < len(all_dis_dates) else "",
                    "source_record_id": anchor.get("source_record_id", ""),
                    "doctor_id": "DR_000000",
                    "surgeries": [
                        {
                            "date": anchor.get("date", ""),
                            "eye": anchor.get("eye", "Unknown"),
                            "name": normalize_surgery_name(anchor.get("name", "Surgery"), anchor.get("eye", "Unknown")),
                        }
                    ],
                    "medications": [],
                    "s1_eyes_data": {"OD": {}, "OS": {}},
                    "s2_eyes_data": {"OD": {}, "OS": {}},
                    "ophthalmic_diagnoses": {
                        "primary_condition": "",
                        "primary_condition_laterality": "Unknown",
                        "primary_condition_stage": "Unknown",
                        "primary_condition_duration": None,
                        "secondary_conditions": [],
                        "current_medications": [],
                    },
                }
            )

    episodes: List[Dict[str, Any]] = []
    for ep_idx, seed in enumerate(episode_seeds):
        ep = merge_template(ep_tpl, {})
        ep["a_treatment_action"]["surgeries"] = []
        ep["a_treatment_action"]["medications"] = []
        ep["follow_up_state"]["followup_records"] = []
        ep["evidence_refs"] = []
        ep["s1_pre_intervention_state"]["evidence_refs"] = []
        ep["a_treatment_action"]["evidence_refs"] = []
        ep["s2_post_surgery_state"]["evidence_refs"] = []

        ep["episode_id"] = f"EP_{pnum:03d}{ep_idx:03d}"
        ep["episode_index"] = ep_idx

        first_surg = seed["surgeries"][0] if seed["surgeries"] else {"date": "", "eye": "Unknown", "name": "Surgery"}
        label_name = re.sub(r"\s+", "_", first_surg.get("name", "episode")).strip("_")[:48]
        ep["episode_label"] = (
            f"{first_surg.get('date', '')}_{first_surg.get('eye', 'Unknown')}_{label_name}"
            if first_surg.get("date", "")
            else f"episode_{ep_idx}"
        )

        adm_date = seed.get("admission_date", "") or first_surg.get("date", "")
        dis_date = seed.get("discharge_date", "")
        ep["hospitalization_info"]["admission_date"] = adm_date
        ep["hospitalization_info"]["discharge_date"] = dis_date
        if adm_date and dis_date:
            try:
                dt0 = datetime.strptime(adm_date, "%Y-%m-%d")
                dt1 = datetime.strptime(dis_date, "%Y-%m-%d")
                ep["hospitalization_info"]["stay_length_days"] = max(0, (dt1 - dt0).days)
            except Exception:  # pylint: disable=broad-except
                ep["hospitalization_info"]["stay_length_days"] = None

        linked = []
        if seed.get("source_record_id", ""):
            linked.append(seed["source_record_id"])
        linked.extend(admission_record_ids)
        linked.extend(discharge_record_ids)
        ep["linked_record_ids"] = sorted(set([x for x in linked if x]))

        ep["ophthalmic_diagnoses"] = merge_template(
            ep_tpl["ophthalmic_diagnoses"],
            seed.get("ophthalmic_diagnoses", {}),
        )

        ep["s1_pre_intervention_state"]["observation_date"] = adm_date
        ep["s1_pre_intervention_state"]["glaucoma_diagnosis_type"] = (
            patient.get("glaucoma_type", "Unknown")
            if patient.get("glaucoma_type", "Unknown") in {"AACG", "CACG", "POAG"}
            else "Unknown"
        )
        for eye in ("OD", "OS"):
            overlay_eye(ep["s1_pre_intervention_state"]["eyes_data"][eye], seed.get("s1_eyes_data", {}).get(eye, {}))

        ep["s2_post_surgery_state"]["observation_date"] = dis_date or first_surg.get("date", "")
        for eye in ("OD", "OS"):
            overlay_eye(ep["s2_post_surgery_state"]["eyes_data"][eye], seed.get("s2_eyes_data", {}).get(eye, {}))

        action_rows = []
        for surg_idx, surg in enumerate(seed.get("surgeries", [])):
            action_rows.append(
                {
                    "action_id": f"ACT_{pnum:03d}{ep_idx * 10 + surg_idx:03d}",
                    "date": surg.get("date", ""),
                    "eye": surg.get("eye", "Unknown"),
                    "name": surg.get("name", ""),
                    "doctor_id": seed.get("doctor_id", "DR_000000"),
                    "source_record_id": seed.get("source_record_id", ""),
                }
            )
        ep["a_treatment_action"]["surgeries"] = action_rows
        med_rows: List[Dict[str, Any]] = []
        for raw_med in seed.get("medications", []):
            med = merge_template(med_tpl, raw_med)
            if med.get("phase", "") not in {"pre_op", "intra_op", "post_op", "discharge", "followup", "chronic", "unknown"}:
                med["phase"] = "unknown"
            if med.get("eye", "") not in {"OD", "OS", "OU", "Unknown"}:
                med["eye"] = "Unknown"
            med["source_record_id"] = med.get("source_record_id", "") or seed.get("source_record_id", "")
            if not isinstance(med.get("name", ""), str):
                med["name"] = ""
            med_rows.append(med)
        ep["a_treatment_action"]["medications"] = dedupe_med_rows(med_rows)

        episodes.append(ep)

    def attach_episode_medications(ep: Dict[str, Any], meds_to_add: List[Dict[str, Any]]) -> None:
        if not meds_to_add:
            return
        merged = ep.get("a_treatment_action", {}).get("medications", []) + meds_to_add
        ep["a_treatment_action"]["medications"] = dedupe_med_rows(merged)
        curr = set(ep.get("ophthalmic_diagnoses", {}).get("current_medications", []))
        for med in meds_to_add:
            name = med.get("name", "")
            if name:
                curr.add(name)
        ep["ophthalmic_diagnoses"]["current_medications"] = sorted(curr)

    def pick_episode_index_for_date(target_date: str) -> int:
        if not episodes:
            return 0
        if not target_date:
            return 0
        ep_idx = 0
        for i, ep in enumerate(episodes):
            adm = ep.get("hospitalization_info", {}).get("admission_date", "")
            surgeries = ep.get("a_treatment_action", {}).get("surgeries", [])
            surg_date = surgeries[0].get("date", "") if surgeries else ""
            anchor = adm or surg_date
            if anchor and anchor <= target_date:
                ep_idx = i
        return ep_idx

    # Admission medication补全：把术前/长期用药并入对应episode。
    for src in records:
        if src.get("record_type") != "admission":
            continue
        rid = src.get("record_id", "")
        src_date = src.get("document_date", "")
        ep_idx = pick_episode_index_for_date(src_date)
        surg_eye = "Unknown"
        if episodes[ep_idx]["a_treatment_action"]["surgeries"]:
            surg_eye = episodes[ep_idx]["a_treatment_action"]["surgeries"][0].get("eye", "Unknown")
        add_meds = extract_admission_medications(
            section=src.get("_raw_text", "") or src.get("text", ""),
            source_record_id=rid,
            surgery_eye=surg_eye,
        )
        attach_episode_medications(episodes[ep_idx], add_meds)
        if rid and rid not in episodes[ep_idx]["linked_record_ids"]:
            episodes[ep_idx]["linked_record_ids"].append(rid)

    # 对于仍然没有药物的episode，回退到关联出院记录再抽一次。
    record_map = {r.get("record_id", ""): r for r in records}
    for ep in episodes:
        if ep.get("a_treatment_action", {}).get("medications"):
            continue
        surg_eye = "Unknown"
        if ep["a_treatment_action"]["surgeries"]:
            surg_eye = ep["a_treatment_action"]["surgeries"][0].get("eye", "Unknown")
        fallback_meds: List[Dict[str, Any]] = []
        for rid in ep.get("linked_record_ids", []):
            rec = record_map.get(rid, {})
            if rec.get("record_type") == "discharge":
                fallback_meds.extend(
                    extract_discharge_medications(
                        section=rec.get("_raw_text", "") or rec.get("text", ""),
                        source_record_id=rid,
                        surgery_eye=surg_eye,
                    )
                )
            if rec.get("record_type") == "admission":
                fallback_meds.extend(
                    extract_admission_medications(
                        section=rec.get("_raw_text", "") or rec.get("text", ""),
                        source_record_id=rid,
                        surgery_eye=surg_eye,
                    )
                )
        attach_episode_medications(ep, fallback_meds)

    surgery_dates = []
    for ep in episodes:
        sd = ""
        if ep["a_treatment_action"]["surgeries"]:
            sd = ep["a_treatment_action"]["surgeries"][0].get("date", "")
        surgery_dates.append(sd)

    followup_records_src = [r for r in records if r.get("record_type") == "followup"]
    followup_records_src.sort(key=lambda x: (x.get("document_date", ""), x.get("record_index", 0)))

    for src in followup_records_src:
        rid = src["record_id"]
        facts = extract_followup_regex_facts(src.get("_raw_text", ""))
        fu = merge_template(fu_tpl, existing_fu_by_record.get(rid, {}))
        fu["source_record_id"] = rid
        fu["followup_date"] = facts.get("followup_date", "") or src.get("document_date", "")
        if not isinstance(fu.get("followup_complications", ""), str):
            fu["followup_complications"] = ""

        for eye in ("OD", "OS"):
            uc = facts[eye].get("ucva", "")
            bc = facts[eye].get("bcva", "")
            iop = facts[eye].get("iop", None)
            cdr = facts[eye].get("cup_to_disc_ratio", None)
            if uc:
                fu["eyes_data"][eye]["ucva"] = uc
            if bc:
                fu["eyes_data"][eye]["bcva"] = bc
            if iop is not None:
                fu["eyes_data"][eye]["iop"] = iop
            if cdr is not None:
                fu["eyes_data"][eye]["cup_to_disc_ratio"] = cdr
            # avoid UCVA/BCVA forced equal when only one value exists
            if fu["eyes_data"][eye].get("ucva", "") == fu["eyes_data"][eye].get("bcva", ""):
                if bc == "" and fu["eyes_data"][eye].get("ucva", ""):
                    fu["eyes_data"][eye]["bcva"] = ""

        ep_idx = 0
        fd = fu.get("followup_date", "")
        if fd:
            for idx, sdate in enumerate(surgery_dates):
                if sdate and sdate <= fd:
                    ep_idx = idx
        surg_eye = "Unknown"
        if episodes[ep_idx]["a_treatment_action"]["surgeries"]:
            surg_eye = episodes[ep_idx]["a_treatment_action"]["surgeries"][0].get("eye", "Unknown")
        fu_meds = extract_followup_medications(
            raw_text=src.get("_raw_text", ""),
            source_record_id=rid,
            surgery_eye=surg_eye,
        )
        if fu_meds:
            attach_episode_medications(episodes[ep_idx], fu_meds)
        episodes[ep_idx]["follow_up_state"]["followup_records"].append(fu)
        if rid not in episodes[ep_idx]["linked_record_ids"]:
            episodes[ep_idx]["linked_record_ids"].append(rid)

    action_counter = 0
    for ep in episodes:
        for surg in ep.get("a_treatment_action", {}).get("surgeries", []):
            surg["action_id"] = f"ACT_{pnum:03d}{action_counter:03d}"
            action_counter += 1
        for med in ep.get("a_treatment_action", {}).get("medications", []):
            med["action_id"] = f"ACT_{pnum:03d}{action_counter:03d}"
            action_counter += 1

    now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    for ep_idx, ep in enumerate(episodes):
        furs = ep["follow_up_state"]["followup_records"]
        furs.sort(key=lambda x: (x.get("followup_date", ""), x.get("source_record_id", "")))
        surg_date = ep["a_treatment_action"]["surgeries"][0].get("date", "") if ep["a_treatment_action"]["surgeries"] else ""
        for fu_idx, fu in enumerate(furs):
            fu["followup_index"] = fu_idx
            fu["followup_id"] = f"FU_{pnum:03d}{ep_idx}{fu_idx:02d}"
            if surg_date and fu.get("followup_date", ""):
                try:
                    dt0 = datetime.strptime(surg_date, "%Y-%m-%d")
                    dt1 = datetime.strptime(fu["followup_date"], "%Y-%m-%d")
                    fu["days_since_intervention"] = max(0, (dt1 - dt0).days)
                except Exception:  # pylint: disable=broad-except
                    fu["days_since_intervention"] = fu.get("days_since_intervention", None)

        st = ep["episode_status_tracking"]
        s1_complete = any(has_eye_content(ep["s1_pre_intervention_state"]["eyes_data"][e]) for e in ("OD", "OS"))
        s2_complete = any(has_eye_content(ep["s2_post_surgery_state"]["eyes_data"][e]) for e in ("OD", "OS"))
        st["s1_pre_intervention_state_status"] = "complete" if s1_complete else "missing"
        st["a_treatment_action_status"] = "complete" if ep["a_treatment_action"]["surgeries"] else "missing"
        st["s2_post_surgery_state_status"] = "complete" if s2_complete else "missing"
        st["episode_status"] = "active" if furs else "closed"
        st["follow_up_state_status"] = "complete" if furs else "missing"
        st["verification_status"] = st.get("verification_status", "") or "machine_checked"
        st["verified_by"] = st.get("verified_by", "") or "llm-pipeline-record-level"
        st["verified_at"] = st.get("verified_at", "") or now_iso
        pieces = [
            st.get("s1_pre_intervention_state_status", "missing"),
            st.get("a_treatment_action_status", "missing"),
            st.get("s2_post_surgery_state_status", "missing"),
            st.get("follow_up_state_status", "missing"),
        ]
        st["data_completeness_score"] = round(sum(1 for p in pieces if p == "complete") / 4.0, 3)

    patient["episodes"] = episodes
    final_json["patients"][0] = patient
    return final_json


def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.update(flatten(v, p))
        return out
    if isinstance(obj, list):
        if not obj:
            out[prefix] = []
            return out
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
        return out
    out[prefix] = obj
    return out


def evaluate(pred: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
    pred_f = flatten(pred)
    gt_f = flatten(gt)
    gt_keys = [k for k in gt_f.keys() if not k.endswith(".last_updated_at") and not k.endswith(".verified_at")]
    shared = [k for k in gt_keys if k in pred_f]
    exact = [k for k in shared if pred_f[k] == gt_f[k]]
    iop_keys = [k for k in gt_keys if k.endswith(".iop")]
    iop_shared = [k for k in iop_keys if k in pred_f]
    iop_exact = [k for k in iop_shared if pred_f[k] == gt_f[k]]

    def r(a: int, b: int) -> float:
        return round((a / b), 4) if b else 0.0

    return {
        "gt_leaf_count": len(gt_keys),
        "pred_leaf_count": len(pred_f),
        "shared_leaf_count": len(shared),
        "path_coverage": r(len(shared), len(gt_keys)),
        "exact_match_count_on_shared": len(exact),
        "exact_match_rate_on_shared": r(len(exact), len(shared)),
        "exact_match_rate_vs_gt_total": r(len(exact), len(gt_keys)),
        "iop_shared_count": len(iop_shared),
        "iop_exact_count": len(iop_exact),
        "iop_exact_rate_on_shared": r(len(iop_exact), len(iop_shared)),
    }


def run_llm(prompt: str, model_path: str, max_new_tokens: int, temperature: float, top_p: float, dtype: str, trust_remote_code: bool) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing packages: transformers accelerate sentencepiece") from exc

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = torch.float16 if (dtype == "auto" and torch.cuda.is_available()) else dtype_map.get(dtype, torch.float32)
    if dtype == "auto" and not torch.cuda.is_available():
        torch_dtype = torch.float32

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    sys_prompt = "You extract glaucoma EHR into strict JSON. Output JSON only."
    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        try:
            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        rendered = sys_prompt + "\n\n" + prompt

    inputs = tok(rendered, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    out_ids = gen[0][prompt_len:]
    return tok.decode(out_ids, skip_special_tokens=True).strip()


def build_repair_prompt(raw_text: str) -> str:
    clipped = raw_text[:60000]
    return (
        "Return one valid JSON object only. No markdown. No explanation.\n"
        "Do not output <think> tags.\n"
        "Fix invalid JSON syntax if needed and keep schema field names unchanged.\n"
        "Input text to repair:\n\n"
        + clipped
    )


def main() -> int:
    args = parse_args()
    patient_dir = Path(args.patient_dir).resolve()
    schema_path = Path(args.schema_path).resolve()
    hint_path = Path(args.schema_hint_path).resolve()
    prompt_path = Path(args.prompt_template).resolve()
    output_path = Path(args.output_path).resolve()
    gt_path = Path(args.ground_truth).resolve() if args.ground_truth else None
    eval_output = Path(args.eval_output).resolve() if args.eval_output else None

    schema = load_json(schema_path)
    hint = load_json(hint_path)
    prompt_template = read_text(prompt_path)

    patient_id, glaucoma_type, records = discover_records(
        patient_dir, max_record_chars=args.max_record_chars
    )
    final_prompt = build_prompt(prompt_template, schema, hint, patient_id, glaucoma_type, records)

    if args.dry_run_no_llm:
        payload = {
            "hospitals": [{"hospital_id": "HP_000000", "hospital_name": ""}],
            "doctors": [],
            "patient_demographics": {},
            "patient_status_tracking": {},
            "episodes": [],
            "evidence_catalog": [],
        }
        raw = ""
    else:
        payload = None
        raw = ""
        last_err: str = ""
        attempts = max(1, args.max_retries)
        record_file_count = 0
        for pth in patient_dir.glob("*.txt"):
            if RECORD_RE.match(pth.name):
                record_file_count += 1
        dynamic_cap = max(1200, int(args.prompt_total_record_chars / max(1, record_file_count)))
        for attempt in range(attempts):
            base_max_chars = max(
                1200,
                args.max_record_chars - attempt * max(0, args.retry_reduce_record_chars),
            )
            attempt_max_chars = min(base_max_chars, dynamic_cap)
            patient_id_try, glaucoma_type_try, records_try = discover_records(
                patient_dir,
                max_record_chars=attempt_max_chars,
            )
            prompt_try = build_prompt(
                prompt_template,
                schema,
                hint,
                patient_id_try,
                glaucoma_type_try,
                records_try,
            )
            if attempt > 0:
                prompt_try += (
                    "\n\nRetry mode: previous attempt failed. "
                    "Output one strict JSON object only."
                )
            try:
                raw_try = run_llm(
                    prompt=prompt_try,
                    model_path=args.model_path,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    dtype=args.dtype,
                    trust_remote_code=args.trust_remote_code,
                )
            except Exception as exc:  # pylint: disable=broad-except
                last_err = f"run_llm attempt {attempt + 1} failed: {exc}"
                if attempt + 1 < attempts:
                    time.sleep(max(0.0, args.retry_sleep_seconds))
                continue

            try:
                payload = parse_json_from_response(raw_try)
                raw = raw_try
                final_prompt = prompt_try
                records = records_try
                break
            except Exception as exc:  # pylint: disable=broad-except
                last_err = f"json_parse attempt {attempt + 1} failed: {exc}"
                repair_prompt = build_repair_prompt(raw_try)
                try:
                    repaired = run_llm(
                        prompt=repair_prompt,
                        model_path=args.model_path,
                        max_new_tokens=min(3072, args.max_new_tokens),
                        temperature=0.0,
                        top_p=1.0,
                        dtype=args.dtype,
                        trust_remote_code=args.trust_remote_code,
                    )
                    payload = parse_json_from_response(repaired)
                    raw = repaired
                    final_prompt = prompt_try + "\n\n[JSON_REPAIR_STEP_APPLIED]"
                    records = records_try
                    break
                except Exception as exc2:  # pylint: disable=broad-except
                    last_err = f"repair_parse attempt {attempt + 1} failed: {exc2}"
                    raw = raw_try
                    final_prompt = prompt_try
                    if attempt + 1 < attempts:
                        time.sleep(max(0.0, args.retry_sleep_seconds))
                    continue

        if payload is None:
            print(
                f"[extract_glaucoma_data] warning: all retries failed, fallback to empty payload. last_error={last_err}",
                file=sys.stderr,
            )
            payload = {
                "hospitals": [{"hospital_id": "HP_000000", "hospital_name": ""}],
                "doctors": [],
                "patient_demographics": {},
                "patient_status_tracking": {},
                "episodes": [],
                "evidence_catalog": [],
            }

    if args.save_rendered_prompt:
        p = Path(args.save_rendered_prompt).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(final_prompt, encoding="utf-8")

    if args.save_raw_response and raw:
        p = Path(args.save_raw_response).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(raw, encoding="utf-8")

    final_json = build_final(schema, payload, patient_id, glaucoma_type, records)
    final_json = apply_record_level_refinement(final_json, schema, records)
    final_json = fill_patient_demographics_from_records(final_json, records)
    final_json = auto_build_evidence(final_json, records)
    warns = validate_refs(final_json)
    write_json(output_path, final_json)

    eval_metrics = None
    if gt_path and gt_path.exists():
        gt = load_json(gt_path)
        eval_metrics = evaluate(final_json, gt)
        if eval_output:
            write_json(
                eval_output,
                {
                    "generated_output": output_path.as_posix(),
                    "ground_truth": gt_path.as_posix(),
                    "metrics": eval_metrics,
                    "warnings": warns,
                },
            )

    p = final_json["patients"][0]
    ep_count = len(p.get("episodes", []))
    fu_count = sum(len(ep.get("follow_up_state", {}).get("followup_records", [])) for ep in p.get("episodes", []))
    print(f"patient_id={p['patient_id']} | episodes={ep_count} | followups={fu_count} | evidence={len(final_json.get('evidence_catalog', []))} | warnings={len(warns)}")
    if eval_metrics:
        print("eval:", json.dumps(eval_metrics, ensure_ascii=False))
    if warns:
        print("validation_warnings:")
        for w in warns:
            print(f"- {w}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[extract_glaucoma_data] failed: {exc}", file=sys.stderr)
        raise
