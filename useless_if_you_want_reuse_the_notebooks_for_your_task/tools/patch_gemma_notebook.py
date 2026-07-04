import json
from pathlib import Path

path = Path('/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/model one per model/gemma_4_26b__single_model_dimension_pipeline_robust.ipynb')
out_path = Path('/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/model one per model/gemma_4_26b__single_model_dimension_pipeline_robust_final.ipynb')

with path.open('r', encoding='utf-8') as f:
    nb = json.load(f)

cell4 = '''MODEL_SPEC = {
    "label": "Gemma 4 26B",
    "model_name": "google/gemma-4-26B-A4B",
    "chat_mode": "manual_gemma",
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.84,
    "max_model_len": 4096,
    "max_new_tokens": 80,
    "max_num_seqs": 8,
    "disable_custom_all_reduce": True,
    "cuda_visible_devices": "1,2",
    "tensor_parallel_size": 2,
}

CUDA_VISIBLE_DEVICES = MODEL_SPEC["cuda_visible_devices"]
TENSOR_PARALLEL_SIZE = int(MODEL_SPEC["tensor_parallel_size"])

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = Path("/home/cbenavent/test/Arcom_rares")

DATASET_PATH = PROJECT_ROOT / "data" / "annotation_working_master_human_2100_seed.csv"
OUTPUT_ROOT = PROJECT_ROOT / "communication_function_outputs" / "per_model_dimension_runs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SMOKE_TEST_ROWS = 20
FULL_RUN_LIMIT = None

SCORE_MIN = 1.00
SCORE_MAX = 5.00
ROUND_DIGITS = 2
CHUNK_SIZE = 500
CONVERGENCE_DRIFT_THRESHOLD = 0.08

# Repeat the resolved runtime values here for clarity in later cells.
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

print("MODEL:", MODEL_SPEC["label"])
print("DATASET_PATH:", DATASET_PATH)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("TENSOR_PARALLEL_SIZE:", TENSOR_PARALLEL_SIZE)
print("MAX_NUM_SEQS:", MODEL_SPEC["max_num_seqs"])
print("MAX_NEW_TOKENS:", MODEL_SPEC["max_new_tokens"])
print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
print("VLLM_WORKER_MULTIPROC_METHOD:", os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", ""))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS", ""))
'''

cell13 = '''def slugify_model_name(value: str) -> str:
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


GEMMA_CHAT_TEMPLATE = "<bos><start_of_turn>user\\n{content}<end_of_turn>\\n<start_of_turn>model\\n"


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
    repaired = re.sub(r",\s*([}\]])", r"\\1", repaired)
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
    reason_match = re.search(r'(?:"reason"|reason)\s*[:=]\s*(null|".*?"|\'.*?\')', compact, flags=re.IGNORECASE)

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
'''

cell21 = '''def inspect_gpu_memory():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        lines = []
        for line in completed.stdout.strip().splitlines():
            idx, name, total, used, free = [part.strip() for part in line.split(",", 4)]
            lines.append({
                "gpu_index": idx,
                "gpu_name": name,
                "total_gb": round(float(total) / 1024.0, 2),
                "used_gb": round(float(used) / 1024.0, 2),
                "free_gb": round(float(free) / 1024.0, 2),
            })
        return pd.DataFrame(lines)
    except Exception as exc:
        print("nvidia-smi inspection failed:", exc)
        return pd.DataFrame()


def build_model_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "trust_remote_code": True,
        "gpu_memory_utilization": float(spec["gpu_memory_utilization"]),
        "dtype": spec["dtype"],
        "max_model_len": int(spec["max_model_len"]),
        "disable_log_stats": True,
    }
    if spec.get("disable_custom_all_reduce", False):
        kwargs["disable_custom_all_reduce"] = True
    if spec.get("max_num_seqs") is not None:
        kwargs["max_num_seqs"] = int(spec["max_num_seqs"])
    return kwargs


def summarise_chunk_convergence(results_df: pd.DataFrame, chunk_size: int = CHUNK_SIZE) -> dict[str, Any]:
    ok_df = results_df[results_df["parse_ok"] == True].copy()
    if ok_df.empty:
        return {
            "chunk_count": 0,
            "convergence_proxy": "no",
            "convergence_ratio": 0.0,
            "last_chunk_mean_drift": None,
        }
    ok_df = ok_df.reset_index(drop=True)
    ok_df["chunk_id"] = ok_df.index // chunk_size
    chunk_means = ok_df.groupby("chunk_id")[SCORE_COLS].mean()
    if len(chunk_means) < 3:
        return {
            "chunk_count": int(len(chunk_means)),
            "convergence_proxy": "not_enough_chunks",
            "convergence_ratio": None,
            "last_chunk_mean_drift": None,
        }
    recent = chunk_means.tail(3)
    drift = float(recent.max().sub(recent.min()).mean())
    convergence_ratio = max(0.0, 1.0 - (drift / 1.0))
    converged = drift <= CONVERGENCE_DRIFT_THRESHOLD
    return {
        "chunk_count": int(len(chunk_means)),
        "convergence_proxy": "yes" if converged else "no",
        "convergence_ratio": round(convergence_ratio, 4),
        "last_chunk_mean_drift": round(drift, 4),
    }


def _extract_generated_text(output) -> tuple[str, str]:
    if output is None:
        return "", "missing generation output"

    generated_items = getattr(output, "outputs", None)
    if generated_items is None:
        return "", "generation object has no outputs attribute"

    if len(generated_items) == 0:
        return "", "empty generation output list"

    first_item = generated_items[0]
    raw_text = getattr(first_item, "text", "") or ""
    if not raw_text.strip():
        return raw_text, "empty generated text"

    return raw_text, ""


def run_dimension_batch(llm, run_df: pd.DataFrame, dimension: str, spec: dict[str, Any]) -> pd.DataFrame:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=int(spec["max_new_tokens"]),
    )
    prompts = [
        build_prompt_for_dimension(dimension, text, llm=llm, chat_mode=spec["chat_mode"])
        for text in run_df["model_input_text"].tolist()
    ]
    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    except TypeError:
        outputs = llm.generate(prompts, sampling_params)

    outputs = list(outputs)
    if len(outputs) != len(run_df):
        print(f"[WARN] output count mismatch for {dimension}: prompts={len(run_df)}, outputs={len(outputs)}")
        if len(outputs) < len(run_df):
            outputs = outputs + [None] * (len(run_df) - len(outputs))
        else:
            outputs = outputs[:len(run_df)]

    rows = []
    for input_row, output in zip(run_df.itertuples(index=False), outputs):
        try:
            raw_text, extraction_error = _extract_generated_text(output)
            if extraction_error:
                parsed = {"score": None, "confidence": 0.0, "reason": None, "parse_source": "failed"}
                parse_ok = False
                parse_error = extraction_error
            else:
                parsed, parse_ok, parse_error = parse_dimension_prediction(raw_text)
        except Exception as exc:
            raw_text = ""
            parsed = {"score": None, "confidence": 0.0, "reason": None, "parse_source": "failed"}
            parse_ok = False
            parse_error = f"row processing failed: {exc}"

        rows.append({
            "row_id": int(getattr(input_row, ROW_ID_COL)) if pd.notna(getattr(input_row, ROW_ID_COL)) else None,
            f"{dimension}__score": parsed["score"],
            f"{dimension}__confidence": parsed["confidence"],
            f"{dimension}__reason": parsed["reason"],
            f"{dimension}__parse_source": parsed.get("parse_source", ""),
            f"{dimension}__raw_output": raw_text,
            f"{dimension}__parse_ok": parse_ok,
            f"{dimension}__parse_error": parse_error,
        })
    return pd.DataFrame(rows)


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


def merge_dimension_batches(run_df: pd.DataFrame, dimension_frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = run_df[[ROW_ID_COL, "model_input_text"]].copy()
    for frame in dimension_frames:
        merged = merged.merge(frame, on="row_id", how="left")

    for dim in SCORE_COLS:
        merged[dim] = merged[f"{dim}__score"]

    def build_combined(row):
        preds = {
            dim: {
                "score": row.get(f"{dim}__score"),
                "confidence": row.get(f"{dim}__confidence"),
                "reason": row.get(f"{dim}__reason"),
            }
            for dim in SCORE_COLS
        }
        return combine_dimension_predictions(preds)

    combined = merged.apply(build_combined, axis=1)
    merged["dominant_dimension"] = combined.map(lambda x: x["dominant_dimension"])
    merged["dominant_dimension_score"] = combined.map(lambda x: x["dominant_dimension_score"])
    merged["confidence"] = combined.map(lambda x: x["confidence"])
    merged["reason"] = combined.map(lambda x: x["reason"])

    parse_cols = [f"{dim}__parse_ok" for dim in SCORE_COLS]
    error_cols = [f"{dim}__parse_error" for dim in SCORE_COLS]
    merged["parse_ok"] = merged[parse_cols].fillna(False).all(axis=1)
    merged["parse_error"] = merged[error_cols].apply(
        lambda row: " | ".join(
            f"{dim}: {err}" for dim, err in zip(SCORE_COLS, row.tolist()) if isinstance(err, str) and err.strip()
        ),
        axis=1,
    )
    recovery_cols = [f"{dim}__parse_source" for dim in SCORE_COLS]
    merged["recovered_with_fallback"] = merged[recovery_cols].apply(
        lambda row: any(str(v).strip() == "regex_fallback" for v in row.tolist()),
        axis=1,
    )
    return merged


def run_model_on_dataframe(spec: dict[str, Any], run_df: pd.DataFrame, run_name: str):
    llm = None
    model_kwargs = build_model_kwargs(spec)
    perf_start = time.perf_counter()
    gpu_before = inspect_gpu_memory()

    try:
        llm = LLM(model=spec["model_name"], **model_kwargs)
        gpu_after_load = inspect_gpu_memory()
        dimension_frames = []
        dimension_diag_rows = []

        for dimension in SCORE_COLS:
            print(f"Running batched pass for dimension: {dimension}")
            try:
                frame = run_dimension_batch(llm, run_df, dimension, spec)
            except Exception as exc:
                print(f"[ERROR] dimension batch failed for {dimension}: {exc}")
                frame = run_df[[ROW_ID_COL]].copy()
                frame[f"{dimension}__score"] = None
                frame[f"{dimension}__confidence"] = 0.0
                frame[f"{dimension}__reason"] = None
                frame[f"{dimension}__parse_source"] = "failed"
                frame[f"{dimension}__raw_output"] = ""
                frame[f"{dimension}__parse_ok"] = False
                frame[f"{dimension}__parse_error"] = f"dimension batch failed: {exc}"

            parse_ok_count = int(frame[f"{dimension}__parse_ok"].fillna(False).sum())
            parse_source_counts = frame[f"{dimension}__parse_source"].fillna("missing").astype(str).value_counts().to_dict()
            dimension_diag_rows.append({
                "dimension": dimension,
                "rows_requested": len(run_df),
                "rows_returned": len(frame),
                "parse_ok_count": parse_ok_count,
                "parse_fail_count": len(frame) - parse_ok_count,
                "parse_ok_rate": round(parse_ok_count / len(frame), 4) if len(frame) else 0.0,
                "empty_output_count": int(frame[f"{dimension}__parse_error"].fillna("").eq("empty generation output list").sum()),
                "missing_output_count": int(frame[f"{dimension}__parse_error"].fillna("").eq("missing generation output").sum()),
                "json_parse_count": int(parse_source_counts.get("json", 0)),
                "regex_fallback_count": int(parse_source_counts.get("regex_fallback", 0)),
                "failed_parse_count": int(parse_source_counts.get("failed", 0)),
            })
            dimension_frames.append(frame)

        merged = merge_dimension_batches(run_df, dimension_frames)
        merged["model_name"] = spec["model_name"]
        merged["model_label"] = spec["label"]
        merged["run_name"] = run_name
        dimension_diag_df = pd.DataFrame(dimension_diag_rows)

    except Exception as exc:
        error_text = f"LLM failed: {exc}"
        print("[ERROR]", error_text)
        gpu_after_load = inspect_gpu_memory()
        merged = run_df[[ROW_ID_COL, "model_input_text"]].copy()
        merged["model_name"] = spec["model_name"]
        merged["model_label"] = spec["label"]
        merged["run_name"] = run_name
        for dim in SCORE_COLS:
            merged[f"{dim}__score"] = None
            merged[f"{dim}__confidence"] = 0.0
            merged[f"{dim}__reason"] = None
            merged[f"{dim}__parse_source"] = "failed"
            merged[f"{dim}__raw_output"] = ""
            merged[f"{dim}__parse_ok"] = False
            merged[f"{dim}__parse_error"] = error_text
            merged[dim] = None
        merged["dominant_dimension"] = ""
        merged["dominant_dimension_score"] = None
        merged["confidence"] = 0.0
        merged["reason"] = error_text
        merged["recovered_with_fallback"] = False
        merged["parse_ok"] = False
        merged["parse_error"] = error_text
        dimension_diag_df = pd.DataFrame([{
            "dimension": "all",
            "rows_requested": len(run_df),
            "rows_returned": len(run_df),
            "parse_ok_count": 0,
            "parse_fail_count": len(run_df),
            "parse_ok_rate": 0.0,
            "empty_output_count": None,
            "missing_output_count": None,
            "json_parse_count": 0,
            "regex_fallback_count": 0,
            "failed_parse_count": len(run_df),
        }])

    finally:
        try:
            if llm is not None:
                del llm
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    perf_end = time.perf_counter()
    gpu_after_cleanup = inspect_gpu_memory()
    perf_seconds = max(0.0, perf_end - perf_start)
    throughput = (len(merged) / perf_seconds) if perf_seconds > 0 else float("inf")
    parse_ok_count = int(merged["parse_ok"].fillna(False).sum())

    diagnostics = {
        "model_label": spec["label"],
        "model_name": spec["model_name"],
        "run_name": run_name,
        "rows_requested": len(run_df),
        "rows_returned": len(merged),
        "parse_ok_count": parse_ok_count,
        "parse_fail_count": len(merged) - parse_ok_count,
        "parse_ok_rate": round(parse_ok_count / len(merged), 4) if len(merged) else 0.0,
        "fallback_recovered_rows": int(merged.get("recovered_with_fallback", pd.Series(dtype=bool)).fillna(False).sum()) if "recovered_with_fallback" in merged.columns else 0,
        "total_seconds": round(perf_seconds, 3),
        "rows_per_second": round(throughput, 3),
        "gpu_before": gpu_before.to_dict(orient="records"),
        "gpu_after_load": gpu_after_load.to_dict(orient="records"),
        "gpu_after_cleanup": gpu_after_cleanup.to_dict(orient="records"),
    }
    return merged, diagnostics, dimension_diag_df


def summarize_parse_errors(results_df: pd.DataFrame) -> pd.DataFrame:
    err = results_df["parse_error"].fillna("").astype(str).str.strip()
    out = err.value_counts(dropna=False).reset_index()
    out.columns = ["parse_error", "count"]
    return out


def summarize_parse_sources(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in SCORE_COLS:
        col = f"{dim}__parse_source"
        if col in results_df.columns:
            counts = results_df[col].fillna("missing").astype(str).value_counts().to_dict()
            for source, count in counts.items():
                rows.append({"dimension": dim, "parse_source": source, "count": int(count)})
    return pd.DataFrame(rows)


def save_model_outputs(results_df: pd.DataFrame, diagnostics: dict[str, Any]) -> dict[str, Path]:
    model_slug = slugify_model_name(diagnostics["model_name"])
    model_dir = OUTPUT_ROOT / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = model_dir / f"{model_slug}__{diagnostics['run_name']}.jsonl"
    full_csv_path = model_dir / f"{model_slug}__{diagnostics['run_name']}_full.csv"
    score_csv_path = model_dir / f"output_dimnesions_scores_{model_slug}.csv"
    diag_csv_path = model_dir / f"{model_slug}__{diagnostics['run_name']}_diagnostics.csv"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in results_df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")

    results_df.to_csv(full_csv_path, index=False)

    score_cols_to_save = [
        "row_id",
        "informativeness",
        "expressiveness",
        "phatic",
        "creativeness_poeticness",
        "dominant_dimension",
        "dominant_dimension_score",
        "confidence",
        "reason",
        "parse_ok",
        "parse_error",
        "recovered_with_fallback",
        "model_name",
        "model_label",
        "run_name",
    ]
    available_score_cols = [c for c in score_cols_to_save if c in results_df.columns]
    results_df.loc[:, available_score_cols].to_csv(score_csv_path, index=False)

    pd.DataFrame([diagnostics]).to_csv(diag_csv_path, index=False)

    return {
        "jsonl_path": jsonl_path,
        "full_csv_path": full_csv_path,
        "score_csv_path": score_csv_path,
        "diag_csv_path": diag_csv_path,
    }
'''

cell23 = '''smoke_results, smoke_diag, smoke_dimension_diag = run_model_on_dataframe(MODEL_SPEC, smoke_df, run_name="smoke_test_20_rows")
smoke_conv = summarise_chunk_convergence(smoke_results)
smoke_diag.update(smoke_conv)
display(smoke_results.head(3))
display(pd.DataFrame([smoke_diag])[[
    "model_label",
    "rows_requested",
    "rows_returned",
    "parse_ok_rate",
    "fallback_recovered_rows",
    "rows_per_second",
    "convergence_proxy",
    "convergence_ratio",
    "last_chunk_mean_drift",
]])
print("Per-dimension smoke diagnostics:")
display(smoke_dimension_diag)
print("Smoke parse sources by dimension:")
display(summarize_parse_sources(smoke_results))
print("Top smoke parse errors:")
display(summarize_parse_errors(smoke_results).head(10))
'''

cell25 = '''full_results, full_diag, full_dimension_diag = run_model_on_dataframe(MODEL_SPEC, sample_df, run_name=f"full_{len(sample_df)}_rows")
full_conv = summarise_chunk_convergence(full_results)
full_diag.update(full_conv)
export_paths = save_model_outputs(full_results, full_diag)
full_diag.update({k: str(v) for k, v in export_paths.items()})

full_diag_df = pd.DataFrame([full_diag])
display(full_diag_df)
display(full_diag_df[[
    "model_label",
    "rows_requested",
    "rows_returned",
    "parse_ok_count",
    "parse_fail_count",
    "parse_ok_rate",
    "fallback_recovered_rows",
    "rows_per_second",
    "convergence_proxy",
    "convergence_ratio",
    "last_chunk_mean_drift",
]])
print("Per-dimension full-run diagnostics:")
display(full_dimension_diag)
print("Full-run parse sources by dimension:")
display(summarize_parse_sources(full_results))
print("Top full-run parse errors:")
display(summarize_parse_errors(full_results).head(20))
display(full_diag_df[["model_label", "score_csv_path", "full_csv_path", "diag_csv_path"]])
'''

for idx, text in [(4, cell4), (13, cell13), (21, cell21), (23, cell23), (25, cell25)]:
    nb['cells'][idx]['source'] = [line + '\n' for line in text.split('\n')]

with out_path.open('w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(out_path)
