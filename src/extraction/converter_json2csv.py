#!/usr/bin/env python3
"""
Compact JSON -> CSV converter for glaucoma intermediate schema.

Output exactly 5 core tables:
1) episodes_master.csv
2) actions.csv
3) observations.csv
4) followups.csv
5) evidence.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


EYE_FIELDS = [
    "iop",
    "ucva",
    "bcva",
    "axial_length",
    "central_corneal_thickness",
    "visual_field_md",
    "cup_to_disc_ratio",
    "rnfl_average_thickness",
    "anterior_chamber_depth",
    "angle_status",
    "slit_lamp_findings",
    "other_findings",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert glaucoma intermediate JSON to 5 compact CSV tables.")
    parser.add_argument("--input-json", required=True, help="Path to intermediate JSON.")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV files.")
    parser.add_argument(
        "--clean-out-dir",
        action="store_true",
        help="Remove existing CSV files in out-dir before writing new outputs.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def patient_code(patient_id: str) -> str:
    m = re.match(r"^PT_(\d{6})$", str(patient_id or ""))
    if m:
        return m.group(1)
    return "000000"


def patient_suffix3(patient_id: str) -> str:
    code = patient_code(patient_id)
    return code[-3:]


def ensure_action_id(raw_id: str, patient_id: str, seq: int) -> str:
    rid = clean_spacing(raw_id or "")
    if re.fullmatch(r"ACT_[0-9]{6}", rid):
        return rid
    return f"ACT_{patient_suffix3(patient_id)}{seq:03d}"


def clean_spacing(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = re.sub(r"\s+", " ", text).strip()
    # Remove broken spaces inside Chinese text chunks, e.g. "屈光 不正" -> "屈光不正".
    out = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", out)
    out = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。；：、！？）】》])", "", out)
    out = re.sub(r"(?<=[（【《])\s+(?=[\u4e00-\u9fff])", "", out)
    out = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\(\)])", "", out)
    out = re.sub(r"(?<=[\(\)])\s+(?=[\u4e00-\u9fff])", "", out)
    out = re.sub(r"\s+\(", "(", out)
    return out


def normalize_text_recursive(v: Any) -> Any:
    if isinstance(v, str):
        return clean_spacing(v)
    if isinstance(v, list):
        return [normalize_text_recursive(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize_text_recursive(x) for k, x in v.items()}
    return v


def to_json_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list, str)):
        return json.dumps(normalize_text_recursive(v), ensure_ascii=False) if isinstance(v, (dict, list)) else clean_spacing(v)
    return str(v)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def convert(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    episodes_master: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    followups: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    action_seq_by_patient: Dict[str, int] = {}
    obs_seq_by_patient: Dict[str, int] = {}

    for p in data.get("patients", []):
        patient_id = p.get("patient_id", "")
        p_suffix = patient_suffix3(patient_id)
        hospital_id = p.get("hospital_id", "")
        glaucoma_type = p.get("glaucoma_type", "")
        demo = p.get("patient_demographics", {})
        p_status = p.get("patient_status_tracking", {})

        for ep in p.get("episodes", []):
            episode_id = ep.get("episode_id", "")
            hosp = ep.get("hospitalization_info", {})
            diag = ep.get("ophthalmic_diagnoses", {})
            s1 = ep.get("s1_pre_intervention_state", {})
            a = ep.get("a_treatment_action", {})
            s2 = ep.get("s2_post_surgery_state", {})
            fu_state = ep.get("follow_up_state", {})
            st = ep.get("episode_status_tracking", {})

            surgeries = a.get("surgeries", [])
            medications = a.get("medications", [])
            fu_list = fu_state.get("followup_records", [])

            episodes_master.append(
                {
                    "episode_id": episode_id,
                    "patient_id": patient_id,
                    "hospital_id": hospital_id,
                    "glaucoma_type": glaucoma_type,
                    "episode_index": ep.get("episode_index", ""),
                    "episode_label": clean_spacing(ep.get("episode_label", "")),
                    "admission_date": hosp.get("admission_date", ""),
                    "discharge_date": hosp.get("discharge_date", ""),
                    "stay_length_days": hosp.get("stay_length_days", ""),
                    "chief_complaint": clean_spacing(hosp.get("chief_complaint", "")),
                    "primary_condition": clean_spacing(diag.get("primary_condition", "")),
                    "primary_condition_laterality": diag.get("primary_condition_laterality", ""),
                    "primary_condition_stage": diag.get("primary_condition_stage", ""),
                    "primary_condition_duration": diag.get("primary_condition_duration", ""),
                    "secondary_conditions_json": to_json_text(diag.get("secondary_conditions", [])),
                    "current_medications_json": to_json_text(diag.get("current_medications", [])),
                    "glaucoma_diagnosis_type_s1": s1.get("glaucoma_diagnosis_type", ""),
                    "linked_record_ids_json": to_json_text(ep.get("linked_record_ids", [])),
                    "episode_evidence_refs_json": to_json_text(ep.get("evidence_refs", [])),
                    "surgery_count": len(surgeries),
                    "medication_count": len(medications),
                    "followup_count": len(fu_list),
                    "episode_status": st.get("episode_status", ""),
                    "s1_pre_intervention_state_status": st.get("s1_pre_intervention_state_status", ""),
                    "a_treatment_action_status": st.get("a_treatment_action_status", ""),
                    "s2_post_surgery_state_status": st.get("s2_post_surgery_state_status", ""),
                    "follow_up_state_status": st.get("follow_up_state_status", ""),
                    "data_completeness_score": st.get("data_completeness_score", ""),
                    "episode_verification_status": clean_spacing(st.get("verification_status", "")),
                    "episode_verified_by": clean_spacing(st.get("verified_by", "")),
                    "episode_verified_at": st.get("verified_at", ""),
                    "patient_extraction_status": clean_spacing(p_status.get("extraction_status", "")),
                    "patient_verification_status": clean_spacing(p_status.get("verification_status", "")),
                    "patient_last_updated_at": p_status.get("last_updated_at", ""),
                    "age_first_seen": demo.get("age_first_seen", ""),
                    "biological_sex": demo.get("biological_sex", ""),
                    "race": demo.get("race", ""),
                    "ethnicity": demo.get("ethnicity", ""),
                    "systemic_comorbidities_json": to_json_text(demo.get("systemic_comorbidities", [])),
                    "family_history_glaucoma": demo.get("family_history_glaucoma", ""),
                }
            )

            intraop = a.get("intraoperative_complications", "")
            for surg in surgeries:
                action_seq_by_patient[patient_id] = action_seq_by_patient.get(patient_id, 0) + 1
                action_short_id = ensure_action_id(
                    surg.get("action_id", ""),
                    patient_id=patient_id,
                    seq=action_seq_by_patient[patient_id],
                )
                actions.append(
                    {
                        "action_id": action_short_id,
                        "action_type": "surgery",
                        "patient_id": patient_id,
                        "episode_id": episode_id,
                        "date": surg.get("date", ""),
                        "phase": "intervention",
                        "eye": surg.get("eye", ""),
                        "name": clean_spacing(surg.get("name", "")),
                        "frequency": "",
                        "purpose": "",
                        "doctor_id": surg.get("doctor_id", ""),
                        "source_record_id": surg.get("source_record_id", ""),
                        "intraoperative_complications": clean_spacing(intraop),
                    }
                )

            for med in medications:
                action_seq_by_patient[patient_id] = action_seq_by_patient.get(patient_id, 0) + 1
                action_short_id = ensure_action_id(
                    med.get("action_id", ""),
                    patient_id=patient_id,
                    seq=action_seq_by_patient[patient_id],
                )
                actions.append(
                    {
                        "action_id": action_short_id,
                        "action_type": "medication",
                        "patient_id": patient_id,
                        "episode_id": episode_id,
                        "date": "",
                        "phase": med.get("phase", ""),
                        "eye": med.get("eye", ""),
                        "name": clean_spacing(med.get("name", "")),
                        "frequency": clean_spacing(med.get("frequency", "")),
                        "purpose": clean_spacing(med.get("purpose", "")),
                        "doctor_id": "",
                        "source_record_id": med.get("source_record_id", ""),
                        "intraoperative_complications": "",
                    }
                )

            for stage_name, stage_obj in (("s1_pre_intervention_state", s1), ("s2_post_surgery_state", s2)):
                obs_date = stage_obj.get("observation_date", "")
                for eye, eye_data in stage_obj.get("eyes_data", {}).items():
                    obs_seq_by_patient[patient_id] = obs_seq_by_patient.get(patient_id, 0) + 1
                    obs_short_id = f"OBS_{p_suffix}{obs_seq_by_patient[patient_id]:03d}"
                    row = {
                        "observation_id": obs_short_id,
                        "source_observation_key": f"{episode_id}:{stage_name}:{eye}",
                        "source_type": "episode_stage",
                        "patient_id": patient_id,
                        "episode_id": episode_id,
                        "followup_id": "",
                        "source_record_id": "",
                        "stage_name": stage_name,
                        "observation_date": obs_date,
                        "followup_index": "",
                        "followup_date": "",
                        "days_since_intervention": "",
                        "eye": eye,
                    }
                    for field in EYE_FIELDS:
                        val = eye_data.get(field, "")
                        row[field] = clean_spacing(val) if isinstance(val, str) else val
                    observations.append(row)

            for fu in fu_list:
                followup_id = fu.get("followup_id", "")
                fu_date = fu.get("followup_date", "")
                followups.append(
                    {
                        "followup_id": followup_id,
                        "patient_id": patient_id,
                        "episode_id": episode_id,
                        "source_record_id": fu.get("source_record_id", ""),
                        "followup_index": fu.get("followup_index", ""),
                        "followup_date": fu_date,
                        "days_since_intervention": fu.get("days_since_intervention", ""),
                        "followup_complications": clean_spacing(fu.get("followup_complications", "")),
                        "evidence_refs_json": to_json_text(fu.get("evidence_refs", [])),
                    }
                )

                for eye, eye_data in fu.get("eyes_data", {}).items():
                    obs_seq_by_patient[patient_id] = obs_seq_by_patient.get(patient_id, 0) + 1
                    obs_short_id = f"OBS_{p_suffix}{obs_seq_by_patient[patient_id]:03d}"
                    row = {
                        "observation_id": obs_short_id,
                        "source_observation_key": f"{followup_id}:{eye}",
                        "source_type": "followup",
                        "patient_id": patient_id,
                        "episode_id": episode_id,
                        "followup_id": followup_id,
                        "source_record_id": fu.get("source_record_id", ""),
                        "stage_name": "follow_up_state",
                        "observation_date": fu_date,
                        "followup_index": fu.get("followup_index", ""),
                        "followup_date": fu_date,
                        "days_since_intervention": fu.get("days_since_intervention", ""),
                        "eye": eye,
                    }
                    for field in EYE_FIELDS:
                        val = eye_data.get(field, "")
                        row[field] = clean_spacing(val) if isinstance(val, str) else val
                    observations.append(row)

    for ev in data.get("evidence_catalog", []):
        evidence.append(
            {
                "evidence_id": ev.get("evidence_id", ""),
                "patient_id": ev.get("patient_id", ""),
                "episode_id": ev.get("episode_id", ""),
                "followup_id": ev.get("followup_id", ""),
                "record_id": ev.get("record_id", ""),
                "field_path": clean_spacing(ev.get("field_path", "")),
                "extracted_value": clean_spacing(ev.get("extracted_value", "")),
                "snippet": clean_spacing(ev.get("snippet", "")),
                "source_path": ev.get("source_path", ""),
                "confidence": ev.get("confidence", ""),
                "extraction_method": clean_spacing(ev.get("extraction_method", "")),
                "annotator": clean_spacing(ev.get("annotator", "")),
                "verifiable": clean_spacing(ev.get("verifiable", "")),
            }
        )

    return {
        "episodes_master": episodes_master,
        "actions": actions,
        "observations": observations,
        "followups": followups,
        "evidence": evidence,
    }


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_out_dir:
        for p in out_dir.glob("*.csv"):
            p.unlink()

    data = read_json(input_json)
    tables = convert(data)

    write_csv(
        out_dir / "episodes.csv",
        tables["episodes_master"],
        [
            "episode_id",
            "patient_id",
            "hospital_id",
            "glaucoma_type",
            "episode_index",
            "episode_label",
            "admission_date",
            "discharge_date",
            "stay_length_days",
            "chief_complaint",
            "primary_condition",
            "primary_condition_laterality",
            "primary_condition_stage",
            "primary_condition_duration",
            "secondary_conditions_json",
            "current_medications_json",
            "glaucoma_diagnosis_type_s1",
            "linked_record_ids_json",
            "episode_evidence_refs_json",
            "surgery_count",
            "medication_count",
            "followup_count",
            "episode_status",
            "s1_pre_intervention_state_status",
            "a_treatment_action_status",
            "s2_post_surgery_state_status",
            "follow_up_state_status",
            "data_completeness_score",
            "episode_verification_status",
            "episode_verified_by",
            "episode_verified_at",
            "patient_extraction_status",
            "patient_verification_status",
            "patient_last_updated_at",
            "age_first_seen",
            "biological_sex",
            "race",
            "ethnicity",
            "systemic_comorbidities_json",
            "family_history_glaucoma",
        ],
    )

    write_csv(
        out_dir / "actions.csv",
        tables["actions"],
        [
            "action_id",
            "action_type",
            "patient_id",
            "episode_id",
            "date",
            "phase",
            "eye",
            "name",
            "frequency",
            "purpose",
            "doctor_id",
            "source_record_id",
            "intraoperative_complications",
        ],
    )

    write_csv(
        out_dir / "observations.csv",
        tables["observations"],
        [
            "observation_id",
            "source_observation_key",
            "source_type",
            "patient_id",
            "episode_id",
            "followup_id",
            "source_record_id",
            "stage_name",
            "observation_date",
            "followup_index",
            "followup_date",
            "days_since_intervention",
            "eye",
        ]
        + EYE_FIELDS,
    )

    write_csv(
        out_dir / "followups.csv",
        tables["followups"],
        [
            "followup_id",
            "patient_id",
            "episode_id",
            "source_record_id",
            "followup_index",
            "followup_date",
            "days_since_intervention",
            "followup_complications",
            "evidence_refs_json",
        ],
    )

    write_csv(
        out_dir / "evidence.csv",
        tables["evidence"],
        [
            "evidence_id",
            "patient_id",
            "episode_id",
            "followup_id",
            "record_id",
            "field_path",
            "extracted_value",
            "snippet",
            "source_path",
            "confidence",
            "extraction_method",
            "annotator",
            "verifiable",
        ],
    )

    print(f"Converted to 5 tables in: {out_dir}")
    for name in ["episodes.csv", "actions.csv", "observations.csv", "followups.csv", "evidence.csv"]:
        p = out_dir / name
        row_count = max(0, len(p.read_text(encoding='utf-8-sig').splitlines()) - 1) if p.exists() else 0
        print(f"- {name}: {row_count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
