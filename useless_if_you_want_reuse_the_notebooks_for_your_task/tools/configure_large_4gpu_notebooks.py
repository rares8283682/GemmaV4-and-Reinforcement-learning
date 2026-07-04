import json
from pathlib import Path


ROOT = Path("/Users/raresolteanu/Desktop/Gliner-Work.Dauphine")


def set_cell_source(nb, idx, source: str) -> None:
    nb["cells"][idx]["source"] = source.splitlines(keepends=True)


def replace_or_fail(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError("Target text not found for replacement.")
    return text.replace(old, new)


def update_llama70() -> None:
    path = ROOT / "model one per model" / "llama_3_1_70b__single_model_dimension_pipeline_robustFinal.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))

    set_cell_source(
        nb,
        2,
        """import os

# Set GPU visibility before importing vLLM / torch-backed modules.
# This notebook is configured for a 4-GPU Llama 3.1 70B run.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"
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
""",
    )

    set_cell_source(
        nb,
        4,
        """MODEL_SPEC = {
    "label": "Llama 3.1 70B",
    "model_name": "meta-llama/Llama-3.1-70B-Instruct",
    "chat_mode": "hf_auto",
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.80,
    "max_model_len": 4096,
    "max_new_tokens": 140,
    "max_num_seqs": 8,
    "disable_custom_all_reduce": True,
    "cuda_visible_devices": "0,1,2,3",
    "tensor_parallel_size": 4,
}

CUDA_VISIBLE_DEVICES = MODEL_SPEC["cuda_visible_devices"]
TENSOR_PARALLEL_SIZE = int(MODEL_SPEC["tensor_parallel_size"])
VISIBLE_GPU_IDS = [gpu_id.strip() for gpu_id in CUDA_VISIBLE_DEVICES.split(",") if gpu_id.strip()]
if len(VISIBLE_GPU_IDS) != TENSOR_PARALLEL_SIZE:
    raise ValueError(f"tensor_parallel_size={TENSOR_PARALLEL_SIZE} must match the number of visible GPUs ({len(VISIBLE_GPU_IDS)}): {CUDA_VISIBLE_DEVICES}")

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
print("DATASET_PATH:", DATASET_PATH)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("VISIBLE_GPU_IDS:", VISIBLE_GPU_IDS)
print("TENSOR_PARALLEL_SIZE:", TENSOR_PARALLEL_SIZE)
print("MAX_NUM_SEQS:", MODEL_SPEC["max_num_seqs"])
print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
print("VLLM_WORKER_MULTIPROC_METHOD:", os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", ""))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS", ""))
""",
    )

    cell21 = "".join(nb["cells"][21]["source"])
    old = """def build_model_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
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
"""
    new = """def build_model_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

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
"""
    cell21 = replace_or_fail(cell21, old, new)
    nb["cells"][21]["source"] = cell21.splitlines(keepends=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {path}")


def update_qwen72() -> None:
    path = ROOT / "model one per model" / "qwen_2_5_72b__single_model_dimension_pipeline_robust.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))

    set_cell_source(
        nb,
        2,
        """import os

# Set GPU visibility before importing vLLM / torch-backed modules.
# This notebook is configured for a 4-GPU Qwen 2.5 72B run.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"
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
    "gpu_memory_utilization": 0.80,
    "max_model_len": 4096,
    "max_new_tokens": 140,
    "max_num_seqs": 12,
    "disable_custom_all_reduce": True,
    "cuda_visible_devices": "0,1,2,3",
    "tensor_parallel_size": 4,
}

CUDA_VISIBLE_DEVICES = MODEL_SPEC["cuda_visible_devices"]
TENSOR_PARALLEL_SIZE = int(MODEL_SPEC["tensor_parallel_size"])
VISIBLE_GPU_IDS = [gpu_id.strip() for gpu_id in CUDA_VISIBLE_DEVICES.split(",") if gpu_id.strip()]
if len(VISIBLE_GPU_IDS) != TENSOR_PARALLEL_SIZE:
    raise ValueError(f"tensor_parallel_size={TENSOR_PARALLEL_SIZE} must match the number of visible GPUs ({len(VISIBLE_GPU_IDS)}): {CUDA_VISIBLE_DEVICES}")

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
print("DATASET_PATH:", DATASET_PATH)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("VISIBLE_GPU_IDS:", VISIBLE_GPU_IDS)
print("TENSOR_PARALLEL_SIZE:", TENSOR_PARALLEL_SIZE)
print("MAX_NUM_SEQS:", MODEL_SPEC["max_num_seqs"])
print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
print("VLLM_WORKER_MULTIPROC_METHOD:", os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", ""))
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS", ""))
""",
    )

    cell21 = "".join(nb["cells"][21]["source"])
    old = """def build_model_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
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
"""
    new = """def build_model_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

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
"""
    cell21 = replace_or_fail(cell21, old, new)
    nb["cells"][21]["source"] = cell21.splitlines(keepends=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {path}")


if __name__ == "__main__":
    update_llama70()
    update_qwen72()
