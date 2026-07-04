import json
from pathlib import Path


SRC = Path("/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/model one per model/llama_3_1_8b__single_model_dimension_pipeline_robust.ipynb")
OUT = Path("/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/model one per model/qwen_2_5_72b__single_model_dimension_pipeline_robust.ipynb")


def set_cell_source(nb, idx, source: str) -> None:
    nb["cells"][idx]["source"] = source.splitlines(keepends=True)


with SRC.open("r", encoding="utf-8") as f:
    nb = json.load(f)


set_cell_source(
    nb,
    0,
    """# Qwen 2.5 72B Single-Model Pipeline, Robust Batched Version

This version mirrors the robust Llama, Gemma, and Mistral notebooks so results stay directly comparable.
It keeps:
- one prompt per dimension
- batched inference
- safer generation-output extraction
- per-dimension failure isolation
- explicit parse diagnostics after smoke and full runs
- safer merge and export logic
""",
)

set_cell_source(
    nb,
    2,
    """import os

# Set GPU visibility before importing vLLM / torch-backed modules.
# This notebook is configured for a 2-GPU Qwen 2.5 72B run.
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"

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
    "label": "Qwen 2.5 72B",
    "model_name": "Qwen/Qwen2.5-72B-Instruct",
    "chat_mode": "hf_auto",
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.78,
    "max_model_len": 4096,
    "max_new_tokens": 180,
    "max_num_seqs": 6,
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
print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
print("VLLM_WORKER_MULTIPROC_METHOD:", os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", ""))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS", ""))
""",
)

with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT}")
