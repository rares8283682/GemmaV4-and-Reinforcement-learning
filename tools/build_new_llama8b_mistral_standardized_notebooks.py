import json
from pathlib import Path


ROOT = Path("/Users/raresolteanu/Desktop/Gliner-Work.Dauphine")
MODEL_DIR = ROOT / "model one per model"

COMMON_CELL_2 = """import os

# Set GPU visibility before importing vLLM / torch-backed modules.
# This notebook supports both 2-GPU and 4-GPU runs.
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


CELL_4_LLAMA = """GPU_PROFILE = "4gpu"  # change to "2gpu" if only two GPUs are available

GPU_PROFILES = {
    "2gpu": {
        "cuda_visible_devices": "0,1",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.80,
        "max_num_seqs": 48,
        "disable_custom_all_reduce": False,
    },
    "4gpu": {
        "cuda_visible_devices": "0,1,2,3",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.76,
        "max_num_seqs": 96,
        "disable_custom_all_reduce": True,
    },
}

BASE_MODEL_SPEC = {
    "label": "Llama 3.1 8B",
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "chat_mode": "hf_auto",
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "max_new_tokens": 80,
    "temperature": 0.0,
    "top_p": 1.0,
}

MODEL_SPEC = {**BASE_MODEL_SPEC, **GPU_PROFILES[GPU_PROFILE]}

CUDA_VISIBLE_DEVICES = MODEL_SPEC["cuda_visible_devices"]
TENSOR_PARALLEL_SIZE = int(MODEL_SPEC["tensor_parallel_size"])
VISIBLE_GPU_IDS = [gpu_id.strip() for gpu_id in CUDA_VISIBLE_DEVICES.split(",") if gpu_id.strip()]
if len(VISIBLE_GPU_IDS) != TENSOR_PARALLEL_SIZE:
    raise ValueError(
        f"tensor_parallel_size={TENSOR_PARALLEL_SIZE} must match the number of visible GPUs "
        f"({len(VISIBLE_GPU_IDS)}): {CUDA_VISIBLE_DEVICES}"
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


CELL_4_MISTRAL = """GPU_PROFILE = "4gpu"  # change to "2gpu" if only two GPUs are available

GPU_PROFILES = {
    "2gpu": {
        "cuda_visible_devices": "1,2",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.82,
        "max_num_seqs": 12,
        "disable_custom_all_reduce": False,
        "min_free_gpu_gb": 60.0,
    },
    "4gpu": {
        "cuda_visible_devices": "0,1,2,3",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.78,
        "max_num_seqs": 20,
        "disable_custom_all_reduce": True,
        "min_free_gpu_gb": 50.0,
    },
}

BASE_MODEL_SPEC = {
    "label": "Mistral Small 24B",
    "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
    "chat_mode": "mistral_chat",
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "max_new_tokens": 80,
    "temperature": 0.0,
    "top_p": 1.0,
    "tokenizer_mode": "mistral",
    "config_format": "mistral",
    "load_format": "mistral",
}

MODEL_SPEC = {**BASE_MODEL_SPEC, **GPU_PROFILES[GPU_PROFILE]}

CUDA_VISIBLE_DEVICES = MODEL_SPEC["cuda_visible_devices"]
TENSOR_PARALLEL_SIZE = int(MODEL_SPEC["tensor_parallel_size"])
VISIBLE_GPU_IDS = [gpu_id.strip() for gpu_id in CUDA_VISIBLE_DEVICES.split(",") if gpu_id.strip()]

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
print("MODEL_NAME:", MODEL_SPEC["model_name"])
print("DATASET_PATH:", DATASET_PATH)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("VISIBLE_GPU_IDS:", VISIBLE_GPU_IDS)
print("TENSOR_PARALLEL_SIZE:", TENSOR_PARALLEL_SIZE)
print("MAX_NUM_SEQS:", MODEL_SPEC["max_num_seqs"])
print("GPU_MEMORY_UTILIZATION:", MODEL_SPEC["gpu_memory_utilization"])
print("MAX_NEW_TOKENS:", MODEL_SPEC["max_new_tokens"])
print("MIN_FREE_GPU_GB:", MODEL_SPEC["min_free_gpu_gb"])
print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
print("VLLM_WORKER_MULTIPROC_METHOD:", os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", ""))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS", ""))
"""


CELL_21_PATCHES = {
    'sampling_params = SamplingParams(\n        temperature=0.0,\n        max_tokens=int(spec["max_new_tokens"]),\n    )':
    'sampling_params = SamplingParams(\n        temperature=float(spec.get("temperature", 0.0)),\n        top_p=float(spec.get("top_p", 1.0)),\n        max_tokens=int(spec["max_new_tokens"]),\n    )',
}


TARGETS = [
    {
        "template": MODEL_DIR / "llama_3_1_8b__single_model_dimension_pipeline_robustFinal.ipynb",
        "output": MODEL_DIR / "llama_3_1_8b__single_model_dimension_pipeline_robust_today.ipynb",
        "cell4": CELL_4_LLAMA,
    },
    {
        "template": MODEL_DIR / "mistral_small_24b__single_model_dimension_pipeline_robust.ipynb",
        "output": MODEL_DIR / "mistral_small_24b__single_model_dimension_pipeline_robust_today.ipynb",
        "cell4": CELL_4_MISTRAL,
    },
]


def to_source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.split("\n")]


def main() -> None:
    for target in TARGETS:
        with target["template"].open("r", encoding="utf-8") as f:
            nb = json.load(f)

        nb["cells"][2]["source"] = to_source_lines(COMMON_CELL_2)
        nb["cells"][4]["source"] = to_source_lines(target["cell4"])

        cell21 = "".join(nb["cells"][21]["source"])
        for old, new in CELL_21_PATCHES.items():
            cell21 = cell21.replace(old, new)
        nb["cells"][21]["source"] = to_source_lines(cell21.rstrip("\n"))

        with target["output"].open("w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(target["output"])


if __name__ == "__main__":
    main()
