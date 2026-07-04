import json
from pathlib import Path

path = Path('/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/model one per model/gemma_4_26b__single_model_dimension_pipeline_robust_final.ipynb')
with path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

cell13 = r"""def slugify_model_name(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def clean_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_model_input_text(row: pd.Series, selected_columns=SELECTED_COLUMNS) -> str:
    parts = []
    for col in selected_columns:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                value = clean_text(value)
                if value and value.lower() not in {"nan", "none"}:
                    parts.append(f"{col}: {value}")
    return "\n".join(parts)


def build_input_text(row: pd.Series) -> str:
    return build_model_input_text(row, SELECTED_COLUMNS)


GEMMA_CHAT_TEMPLATE = "<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"


def build_dimension_prompt_content(dimension: str, ad_text: str) -> str:
    clean_ad_text = clean_text(ad_text)
    return "\n\n".join([
        COMMON_SYSTEM_PROMPT.strip(),
        DIMENSION_INSTRUCTIONS[dimension].strip(),
        "AD TO SCORE",
        clean_ad_text,
        "Return only strict JSON.",
    ])


def render_prompt_for_model(content: str, llm=None, chat_mode: str = "hf_auto") -> str:
    if chat_mode == "manual_gemma":
        return GEMMA_CHAT_TEMPLATE.format(content=content)
    if chat_mode == "plain" or llm is None:
        return content
    try:
        tokenizer = llm.get_tokenizer()
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        print(f"[WARN] apply_chat_template failed, falling back to plain prompt: {exc}")
        return content


def build_prompt_for_dimension(dimension: str, input_text: str, llm=None, chat_mode: str = "hf_auto") -> str:
    content = build_dimension_prompt_content(dimension, input_text)
    return render_prompt_for_model(content, llm=llm, chat_mode=chat_mode)


def clamp_score(value: Any, default: float = 1.0) -> float:
    try:
        text = str(value).strip().replace(",", ".")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        parsed = float(match.group(0)) if match else float(default)
        score = round(parsed, ROUND_DIGITS)
    except Exception:
        score = float(default)
    score = max(SCORE_MIN, min(SCORE_MAX, score))
    return round(score, ROUND_DIGITS)


def _repair_json_candidate(text: str) -> str:
    repaired = str(text)
    repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = repaired.strip()
    return repaired


def extract_json_object(text: str) -> dict:
    text = str(text or "")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)

    brace_matches = list(re.finditer(r"\{", text))
    for match in brace_matches[:8]:
        candidates.append(text[match.start():])

    decoder = json.JSONDecoder()
    for candidate in candidates:
        repaired = _repair_json_candidate(candidate)
        for snippet in (candidate, repaired):
            for match in re.finditer(r"\{", snippet):
                try:
                    payload, _ = decoder.raw_decode(snippet[match.start():].strip())
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    continue
    raise ValueError("Could not extract JSON from model output.")


def extract_dimension_fields_from_text(raw_text: str) -> dict:
    compact = " ".join(str(raw_text or "").split())

    score_match = re.search(r'(?:"score"|score)\s*[:=]\s*"?([1-5](?:[\.,]\d+)?)"?', compact, flags=re.IGNORECASE)
    confidence_match = re.search(r'(?:"confidence"|confidence)\s*[:=]\s*"?([01](?:[\.,]\d+)?)"?', compact, flags=re.IGNORECASE)
    reason_match = re.search(r"(?:\"reason\"|reason)\s*[:=]\s*(null|\".*?\"|'.*?')", compact, flags=re.IGNORECASE)

    parsed = {
        "score": clamp_score(score_match.group(1)) if score_match else None,
        "confidence": None,
        "reason": None,
    }

    if confidence_match:
        try:
            parsed["confidence"] = round(float(confidence_match.group(1).replace(",", ".")), ROUND_DIGITS)
            parsed["confidence"] = max(0.0, min(1.0, parsed["confidence"]))
        except Exception:
            parsed["confidence"] = None

    if reason_match:
        raw_reason = reason_match.group(1)
        if raw_reason.lower() == "null":
            parsed["reason"] = None
        else:
            parsed["reason"] = clean_text(raw_reason.strip('"\'')) or None

    return parsed


def parse_dimension_prediction(raw_text: str) -> tuple[dict, bool, str]:
    errors = []
    parsed = {"score": None, "confidence": None, "reason": None, "parse_source": ""}

    try:
        payload = extract_json_object(raw_text)
        parsed["score"] = clamp_score(payload.get("score", 1.0))
        try:
            confidence = round(float(payload.get("confidence", 0.5)), ROUND_DIGITS)
        except Exception:
            confidence = 0.5
        parsed["confidence"] = max(0.0, min(1.0, confidence))
        reason = payload.get("reason", None)
        if reason is None or (isinstance(reason, float) and pd.isna(reason)):
            parsed["reason"] = None
        else:
            parsed["reason"] = clean_text(reason) or None
        parsed["parse_source"] = "json"
    except Exception as exc:
        errors.append(str(exc))

    if parsed["score"] is None:
        fallback = extract_dimension_fields_from_text(raw_text)
        if fallback["score"] is not None:
            parsed["score"] = fallback["score"]
            parsed["confidence"] = fallback["confidence"] if fallback["confidence"] is not None else 0.5
            parsed["reason"] = fallback["reason"]
            parsed["parse_source"] = "regex_fallback"

    parse_ok = parsed["score"] is not None
    if not parse_ok:
        return {"score": None, "confidence": 0.0, "reason": None, "parse_source": "failed"}, False, " | ".join(errors) or "could not parse dimension output"

    if parsed["confidence"] is None:
        parsed["confidence"] = 0.5

    if not parsed["parse_source"]:
        parsed["parse_source"] = "json"

    return parsed, True, ""


def combine_dimension_predictions(preds: dict[str, dict]) -> dict:
    scores = {dim: preds[dim]["score"] for dim in SCORE_COLS}
    valid_scores = {k: v for k, v in scores.items() if v is not None and pd.notna(v)}

    if valid_scores:
        ranked = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)
        top_score = ranked[0][1]
        top_labels = [k for k, v in ranked if v == top_score]
        dominant_dimension = "mixed" if len(top_labels) > 1 else top_labels[0]
        dominant_dimension_score = round(top_score, ROUND_DIGITS)
    else:
        dominant_dimension = ""
        dominant_dimension_score = None

    confidence_values = [
        preds[dim]["confidence"]
        for dim in SCORE_COLS
        if preds[dim]["score"] is not None and pd.notna(preds[dim]["score"])
    ]
    confidence = round(float(np.mean(confidence_values)), ROUND_DIGITS) if confidence_values else 0.0

    reason_parts = [f"{dim}: {preds[dim]['reason']}" for dim in SCORE_COLS if preds[dim]["reason"]]
    reason = " | ".join(reason_parts[:2]) if reason_parts else None

    return {
        **scores,
        "dominant_dimension": dominant_dimension,
        "dominant_dimension_score": dominant_dimension_score,
        "confidence": confidence,
        "reason": reason,
    }


def default_prediction(reason: str, parse_ok: bool = False) -> dict:
    return {
        "prediction": {
            "informativeness": None,
            "expressiveness": None,
            "phatic": None,
            "creativeness_poeticness": None,
            "dominant_dimension": "",
            "dominant_dimension_score": None,
            "confidence": 0.0,
            "reason": reason,
        },
        "dimension_outputs": {dim: {"score": None, "confidence": 0.0, "reason": None, "raw_output": "", "parse_source": "failed"} for dim in SCORE_COLS},
        "dimension_parse": {dim: False for dim in SCORE_COLS},
        "parse_ok": parse_ok,
        "parse_error": "" if parse_ok else reason,
    }
"""

nb['cells'][13]['source'] = [line + '\n' for line in cell13.split('\n')]
with path.open('w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(path)
