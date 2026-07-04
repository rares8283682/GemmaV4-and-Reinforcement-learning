import json
from pathlib import Path


SRC = Path("/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/model one per model/llama_3_1_8b__single_model_dimension_pipeline_robustFinal.ipynb")
OUT = Path("/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/model one per model/llama_3_1_70b_quantized__single_model_dimension_pipeline_robust.ipynb")


def set_cell_source(nb, idx, source: str) -> None:
    nb["cells"][idx]["source"] = source.splitlines(keepends=True)


def replace_in_cell(nb, idx, old: str, new: str) -> None:
    src = "".join(nb["cells"][idx]["source"])
    if old not in src:
        raise ValueError(f"Could not find target text in cell {idx!r}")
    nb["cells"][idx]["source"] = src.replace(old, new).splitlines(keepends=True)


with SRC.open("r", encoding="utf-8") as f:
    nb = json.load(f)


set_cell_source(
    nb,
    0,
    """# Llama 3.1 70B Quantized Single-Model Pipeline, Robust Batched Version

This version is optimized for a 1-GPU quantized Llama 70B run while preserving
the same structure as the other robust notebooks so the outputs remain directly comparable.
It keeps:
- one prompt per dimension
- batched inference
- safer generation-output extraction
- per-dimension failure isolation
- explicit parse diagnostics after smoke and full runs
- safer merge and export logic
- cleaner output naming via a fixed output slug
""",
)

set_cell_source(
    nb,
    2,
    """import os

# Set GPU visibility before importing vLLM / torch-backed modules.
# This notebook is configured for a 1-GPU quantized Llama 70B run.
# Change only the GPU id if your available GPU is different.
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_USE_FASTOKENS"] = "1"

import gc
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

# Ensure vLLM child processes load the conda libstdc++ (fixes GLIBCXX_3.4.31 errors).
CONDA_LIBSTDCXX = Path("/home/cbenavent/envs/vllm312/lib/libstdc++.so.6")
if CONDA_LIBSTDCXX.exists():
    os.environ["LD_PRELOAD"] = str(CONDA_LIBSTDCXX)
    existing_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    conda_lib_dir = str(CONDA_LIBSTDCXX.parent)
    if conda_lib_dir not in existing_ld_library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib_dir}:{existing_ld_library_path}" if existing_ld_library_path else conda_lib_dir

import numpy as np
import pandas as pd
from IPython.display import display
from vllm import LLM, SamplingParams
""",
)

set_cell_source(
    nb,
    4,
    """MODEL_SPEC = {
    "label": "Llama 3.1 70B",
    "model_name": "casperhansen/llama-3.1-70b-instruct-awq",
    "output_slug": "llama_3_1_70b_quantized",
    "chat_mode": "hf_auto",
    "dtype": "float16",
    "quantization": "awq",
    "kv_cache_dtype": "fp8",
    "gpu_memory_utilization": 0.96,
    "max_model_len": 3072,
    "max_new_tokens": 90,
    "max_num_seqs": 24,
    "enable_prefix_caching": True,
    "enforce_eager": False,
    "disable_custom_all_reduce": False,
    "cuda_visible_devices": "2",
    "tensor_parallel_size": 1,
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

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

print("MODEL:", MODEL_SPEC["label"])
print("HF MODEL:", MODEL_SPEC["model_name"])
print("OUTPUT_SLUG:", MODEL_SPEC["output_slug"])
print("DATASET_PATH:", DATASET_PATH)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("TENSOR_PARALLEL_SIZE:", TENSOR_PARALLEL_SIZE)
print("MAX_NUM_SEQS:", MODEL_SPEC["max_num_seqs"])
print("MAX_MODEL_LEN:", MODEL_SPEC["max_model_len"])
print("KV_CACHE_DTYPE:", MODEL_SPEC["kv_cache_dtype"])
print("QUANTIZATION:", MODEL_SPEC["quantization"])
""",
)

replace_in_cell(
    nb,
    21,
    """def build_model_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
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
""",
    """def build_model_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
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
    if spec.get("quantization"):
        kwargs["quantization"] = spec["quantization"]
    if spec.get("kv_cache_dtype"):
        kwargs["kv_cache_dtype"] = spec["kv_cache_dtype"]
    if spec.get("enable_prefix_caching") is not None:
        kwargs["enable_prefix_caching"] = bool(spec["enable_prefix_caching"])
    if spec.get("enforce_eager") is not None:
        kwargs["enforce_eager"] = bool(spec["enforce_eager"])
    return kwargs
""",
)

replace_in_cell(
    nb,
    21,
    """def save_model_outputs(results_df: pd.DataFrame, diagnostics: dict[str, Any]) -> dict[str, Path]:
    model_slug = slugify_model_name(diagnostics["model_name"])
    model_dir = OUTPUT_ROOT / model_slug
""",
    """def save_model_outputs(results_df: pd.DataFrame, diagnostics: dict[str, Any]) -> dict[str, Path]:
    model_slug = slugify_model_name(MODEL_SPEC.get("output_slug") or diagnostics["model_label"])
    model_dir = OUTPUT_ROOT / model_slug
""",
)


with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT}")
