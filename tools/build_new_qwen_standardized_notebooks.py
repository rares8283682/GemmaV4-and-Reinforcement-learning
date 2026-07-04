import json
from pathlib import Path


ROOT = Path("/Users/raresolteanu/Desktop/Gliner-Work.Dauphine")
MODEL_DIR = ROOT / "model one per model"
TEMPLATE_PATH = MODEL_DIR / "qwen_2_5_72b__single_model_dimension_pipeline_robustFinalFinalF.ipynb"


COMMON_CELL_2 = """import os

# Set GPU visibility before importing vLLM / torch-backed modules.
# This notebook supports both 2-GPU and 4-GPU BF16 Qwen runs.
# Change GPU_PROFILE in the config cell, then restart the kernel.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
"""


CELL_4_TEMPLATE = """GPU_PROFILE = "4gpu"  # change to "2gpu" if only two GPUs are available

GPU_PROFILES = {{
    "2gpu": {{
        "cuda_visible_devices": "0,1",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": {gpu_mem_2},
        "max_num_seqs": {max_num_seqs_2},
        "disable_custom_all_reduce": False,
    }},
    "4gpu": {{
        "cuda_visible_devices": "0,1,2,3",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": {gpu_mem_4},
        "max_num_seqs": {max_num_seqs_4},
        "disable_custom_all_reduce": True,
    }},
}}

BASE_MODEL_SPEC = {{
    "label": "{label}",
    "model_name": "{model_name}",
    "chat_mode": "hf_auto",
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "max_new_tokens": 80,
    "temperature": 0.0,
    "top_p": 1.0,
}}

MODEL_SPEC = {{**BASE_MODEL_SPEC, **GPU_PROFILES[GPU_PROFILE]}}

CUDA_VISIBLE_DEVICES = MODEL_SPEC["cuda_visible_devices"]
TENSOR_PARALLEL_SIZE = int(MODEL_SPEC["tensor_parallel_size"])
VISIBLE_GPU_IDS = [gpu_id.strip() for gpu_id in CUDA_VISIBLE_DEVICES.split(",") if gpu_id.strip()]
if len(VISIBLE_GPU_IDS) != TENSOR_PARALLEL_SIZE:
    raise ValueError(
        f"tensor_parallel_size={{TENSOR_PARALLEL_SIZE}} must match the number of visible GPUs "
        f"({{len(VISIBLE_GPU_IDS)}}): {{CUDA_VISIBLE_DEVICES}}"
    )

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

print("GPU_PROFILE:", GPU_PROFILE)
print("MODEL:", MODEL_SPEC["label"])
print("DATASET_PATH:", DATASET_PATH)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("VISIBLE_GPU_IDS:", VISIBLE_GPU_IDS)
print("TENSOR_PARALLEL_SIZE:", TENSOR_PARALLEL_SIZE)
print("MAX_NUM_SEQS:", MODEL_SPEC["max_num_seqs"])
print("MAX_NEW_TOKENS:", MODEL_SPEC["max_new_tokens"])
print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
print("VLLM_WORKER_MULTIPROC_METHOD:", os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", ""))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS", ""))
"""


CELL_21_PATCHES = {
    'if spec.get("max_num_seqs") is not None:\n        kwargs["max_num_seqs"] = int(spec["max_num_seqs"])\n    return kwargs':
    'if spec.get("max_num_seqs") is not None:\n        kwargs["max_num_seqs"] = int(spec["max_num_seqs"])\n    return kwargs',
    'sampling_params = SamplingParams(\n        temperature=0.0,\n        max_tokens=int(spec["max_new_tokens"]),\n    )':
    'sampling_params = SamplingParams(\n        temperature=float(spec.get("temperature", 0.0)),\n        top_p=float(spec.get("top_p", 1.0)),\n        max_tokens=int(spec["max_new_tokens"]),\n    )',
}


TARGETS = [
    {
        "filename": "qwen_2_5_14b__single_model_dimension_pipeline_robust_today.ipynb",
        "label": "Qwen 2.5 14B",
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "gpu_mem_2": "0.88",
        "max_num_seqs_2": "24",
        "gpu_mem_4": "0.86",
        "max_num_seqs_4": "48",
    },
    {
        "filename": "qwen_2_5_32b__single_model_dimension_pipeline_robust_today.ipynb",
        "label": "Qwen 2.5 32B",
        "model_name": "Qwen/Qwen2.5-32B-Instruct",
        "gpu_mem_2": "0.84",
        "max_num_seqs_2": "16",
        "gpu_mem_4": "0.82",
        "max_num_seqs_4": "32",
    },
]


def to_source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.split("\n")]


def main() -> None:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        template = json.load(f)

    for target in TARGETS:
        nb = json.loads(json.dumps(template))
        nb["cells"][2]["source"] = to_source_lines(COMMON_CELL_2)
        nb["cells"][4]["source"] = to_source_lines(CELL_4_TEMPLATE.format(**target))

        cell21 = "".join(nb["cells"][21]["source"])
        for old, new in CELL_21_PATCHES.items():
            cell21 = cell21.replace(old, new)
        nb["cells"][21]["source"] = to_source_lines(cell21.rstrip("\n"))

        out_path = MODEL_DIR / target["filename"]
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(out_path)


if __name__ == "__main__":
    main()
