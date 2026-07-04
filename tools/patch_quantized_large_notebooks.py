import json
from pathlib import Path

ROOT = Path('/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/model one per model')

NOTEBOOKS = {
    ROOT / 'llama_3_1_70b_quantized__single_model_dimension_pipeline_robust.ipynb': {
        'cell2': '''import os

# Set GPU visibility before importing vLLM / torch-backed modules.
# This notebook supports both 2-GPU and 4-GPU quantized Llama runs.
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
''',
        'cell4': '''GPU_PROFILE = "4gpu"  # change to "2gpu" if only two GPUs are available

GPU_PROFILES = {
    "2gpu": {
        "cuda_visible_devices": "0,1",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 16,
        "disable_custom_all_reduce": False,
    },
    "4gpu": {
        "cuda_visible_devices": "0,1,2,3",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 32,
        "disable_custom_all_reduce": True,
    },
}

BASE_MODEL_SPEC = {
    "label": "Llama 3.3 70B",
    "model_name": "casperhansen/llama-3.3-70b-instruct-awq",
    "output_slug": "llama_3_3_70b_quantized",
    "chat_mode": "hf_auto",
    "dtype": "float16",
    "quantization": "awq_marlin",
    "kv_cache_dtype": "auto",
    "max_model_len": 4096,
    "max_new_tokens": 40,
    "temperature": 0.0,
    "top_p": 1.0,
    "enable_prefix_caching": True,
    "enforce_eager": False,
}

MODEL_SPEC = {**BASE_MODEL_SPEC, **GPU_PROFILES[GPU_PROFILE]}

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

print("GPU_PROFILE:", GPU_PROFILE)
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
''',
    },
    ROOT / 'qwen_2_5_72b_quantized__single_model_dimension_pipeline_robust.ipynb': {
        'cell2': '''import os

# Set GPU visibility before importing vLLM / torch-backed modules.
# This notebook supports both 2-GPU and 4-GPU quantized Qwen runs.
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
import vllm
from IPython.display import display
from vllm import LLM, SamplingParams
''',
        'cell4': '''GPU_PROFILE = "4gpu"  # change to "2gpu" if only two GPUs are available

GPU_PROFILES = {
    "2gpu": {
        "cuda_visible_devices": "0,1",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 16,
        "disable_custom_all_reduce": False,
    },
    "4gpu": {
        "cuda_visible_devices": "0,1,2,3",
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 32,
        "disable_custom_all_reduce": True,
    },
}

BASE_MODEL_SPEC = {
    "label": "Qwen 2.5 72B",
    "model_name": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "output_slug": "qwen_2_5_72b_quantized",
    "chat_mode": "hf_auto",
    "dtype": "float16",
    "quantization": "awq_marlin",
    "kv_cache_dtype": "auto",
    "max_model_len": 4096,
    "max_new_tokens": 40,
    "temperature": 0.0,
    "top_p": 1.0,
    "enable_prefix_caching": True,
    "enforce_eager": False,
}

MODEL_SPEC = {**BASE_MODEL_SPEC, **GPU_PROFILES[GPU_PROFILE]}

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

print("GPU_PROFILE:", GPU_PROFILE)
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
''',
    },
}

for path, edits in NOTEBOOKS.items():
    with path.open('r', encoding='utf-8') as f:
        nb = json.load(f)
    nb['cells'][2]['source'] = [line + '\n' for line in edits['cell2'].split('\n')]
    nb['cells'][4]['source'] = [line + '\n' for line in edits['cell4'].split('\n')]
    with path.open('w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(path)
