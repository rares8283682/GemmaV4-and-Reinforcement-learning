import json
from pathlib import Path


SRC = Path("/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/model one per model/llama_3_1_8b__single_model_dimension_pipeline_batched.ipynb")
OUT = Path("/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/model one per model/llama_3_1_8b__single_model_dimension_pipeline_robust.ipynb")


def set_cell_source(nb, idx, source: str) -> None:
    nb["cells"][idx]["source"] = source.splitlines(keepends=True)


with SRC.open("r", encoding="utf-8") as f:
    nb = json.load(f)


set_cell_source(
    nb,
    0,
    """# Llama 3.1 8B Single-Model Pipeline, Robust Batched Version

This version keeps one prompt per dimension and batched inference, but adds:
- safer generation-output extraction
- per-dimension failure isolation
- explicit parse diagnostics after smoke and full runs
- safer merge logic that does not collapse the whole run because of one malformed output
""",
)

set_cell_source(
    nb,
    21,
    """def inspect_gpu_memory():
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
                parsed = {"score": None, "confidence": 0.0, "reason": None}
                parse_ok = False
                parse_error = extraction_error
            else:
                parsed, parse_ok, parse_error = parse_dimension_prediction(raw_text)
        except Exception as exc:
            raw_text = ""
            parsed = {"score": None, "confidence": 0.0, "reason": None}
            parse_ok = False
            parse_error = f"row processing failed: {exc}"

        rows.append({
            "row_id": int(getattr(input_row, ROW_ID_COL)) if pd.notna(getattr(input_row, ROW_ID_COL)) else None,
            f"{dimension}__score": parsed["score"],
            f"{dimension}__confidence": parsed["confidence"],
            f"{dimension}__reason": parsed["reason"],
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
                frame[f"{dimension}__raw_output"] = ""
                frame[f"{dimension}__parse_ok"] = False
                frame[f"{dimension}__parse_error"] = f"dimension batch failed: {exc}"

            parse_ok_count = int(frame[f"{dimension}__parse_ok"].fillna(False).sum())
            dimension_diag_rows.append({
                "dimension": dimension,
                "rows_requested": len(run_df),
                "rows_returned": len(frame),
                "parse_ok_count": parse_ok_count,
                "parse_fail_count": len(frame) - parse_ok_count,
                "parse_ok_rate": round(parse_ok_count / len(frame), 4) if len(frame) else 0.0,
                "empty_output_count": int(frame[f"{dimension}__parse_error"].fillna("").eq("empty generation output list").sum()),
                "missing_output_count": int(frame[f"{dimension}__parse_error"].fillna("").eq("missing generation output").sum()),
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
            merged[f"{dim}__raw_output"] = ""
            merged[f"{dim}__parse_ok"] = False
            merged[f"{dim}__parse_error"] = error_text
            merged[dim] = None
        merged["dominant_dimension"] = ""
        merged["dominant_dimension_score"] = None
        merged["confidence"] = 0.0
        merged["reason"] = error_text
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
""",
)

set_cell_source(
    nb,
    23,
    """smoke_results, smoke_diag, smoke_dimension_diag = run_model_on_dataframe(MODEL_SPEC, smoke_df, run_name="smoke_test_20_rows")
smoke_conv = summarise_chunk_convergence(smoke_results)
smoke_diag.update(smoke_conv)
display(smoke_results.head(3))
display(pd.DataFrame([smoke_diag])[[
    "model_label",
    "rows_requested",
    "rows_returned",
    "parse_ok_rate",
    "rows_per_second",
    "convergence_proxy",
    "convergence_ratio",
    "last_chunk_mean_drift",
]])
print("Per-dimension smoke diagnostics:")
display(smoke_dimension_diag)
print("Top smoke parse errors:")
display(summarize_parse_errors(smoke_results).head(10))
""",
)

set_cell_source(
    nb,
    25,
    """full_results, full_diag, full_dimension_diag = run_model_on_dataframe(MODEL_SPEC, sample_df, run_name=f"full_{len(sample_df)}_rows")
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
    "rows_per_second",
    "convergence_proxy",
    "convergence_ratio",
    "last_chunk_mean_drift",
]])
print("Per-dimension full-run diagnostics:")
display(full_dimension_diag)
print("Top full-run parse errors:")
display(summarize_parse_errors(full_results).head(20))
display(full_diag_df[["model_label", "score_csv_path", "full_csv_path", "diag_csv_path"]])
""",
)

with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT}")
