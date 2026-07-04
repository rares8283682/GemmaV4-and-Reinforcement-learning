from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/Users/raresolteanu/Desktop/Gliner-Work.Dauphine")
NOTEBOOK_PATH = ROOT / "analysis_notebooks" / "full_model_comparison_analysis.ipynb"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text,
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


cells: list[dict] = []

cells.append(
    md_cell(
        "# Full Model Comparison Analysis\n"
        "\n"
        "This notebook loads all currently available communication-function output folders, builds aligned cross-run tables, and generates polished comparison figures for:\n"
        "\n"
        "- parsing success\n"
        "- runtime throughput\n"
        "- convergence / late-run drift\n"
        "- mean score profiles\n"
        "- dominant-dimension shares\n"
        "- pairwise correlation matrices\n"
        "- exact and within-1 agreement matrices\n"
        "- standard vs rerun vs regex vs quantized comparisons\n"
        "\n"
        "It uses every model-data folder currently present in the repository, so you can compare both the final preferred runs and the earlier variants."
    )
)

cells.append(
    code_cell(
        "from pathlib import Path\n"
        "import json\n"
        "import math\n"
        "import warnings\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "import matplotlib.pyplot as plt\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "pd.set_option('display.max_columns', 200)\n"
        "pd.set_option('display.width', 220)\n"
    )
)

cells.append(
    code_cell(
        "ROOT = Path('/Users/raresolteanu/Desktop/Gliner-Work.Dauphine')\n"
        "DATA_DIR = ROOT / 'model one per model' / 'data'\n"
        "OUT_DIR = ROOT / 'analysis_notebooks' / 'model_comparison_outputs'\n"
        "FIGURES_DIR = OUT_DIR / 'figures'\n"
        "TABLES_DIR = OUT_DIR / 'tables'\n"
        "FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "TABLES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "SCORE_COLS = ['informativeness', 'expressiveness', 'phatic', 'creativeness_poeticness']\n"
        "DOM_ORDER = ['informativeness', 'expressiveness', 'phatic', 'creativeness_poeticness', 'mixed']\n"
        "\n"
        "# Every current output folder is included as its own run variant.\n"
        "MODEL_SPECS = [\n"
        "    {\n"
        "        'label': 'Gemma 4 26B (Original)',\n"
        "        'family': 'standard',\n"
        "        'variant': 'original',\n"
        "        'size_b': 26,\n"
        "        'base_model': 'Gemma 4 26B',\n"
        "        'preferred': False,\n"
        "        'notes': 'Original strict-JSON Gemma run',\n"
        "        'folder': 'GemmaV4',\n"
        "        'full': 'google_gemma_4_26b_a4b__full_8938_rows_full.csv',\n"
        "        'diag': 'google_gemma_4_26b_a4b__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Gemma 4 26B (Regex Rerun)',\n"
        "        'family': 'standard',\n"
        "        'variant': 'regex_rerun',\n"
        "        'size_b': 26,\n"
        "        'base_model': 'Gemma 4 26B',\n"
        "        'preferred': True,\n"
        "        'notes': 'Regex-enhanced Gemma rerun',\n"
        "        'folder': 'GemmaV4_REGEX',\n"
        "        'full': 'google_gemma_4_26b_a4b__full_8938_rows_full.csv',\n"
        "        'diag': 'google_gemma_4_26b_a4b__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Llama 3.1 8B (Original)',\n"
        "        'family': 'standard',\n"
        "        'variant': 'original',\n"
        "        'size_b': 8,\n"
        "        'base_model': 'Llama 3.1 8B',\n"
        "        'preferred': False,\n"
        "        'notes': 'Original Llama 8B run',\n"
        "        'folder': 'Llama8B',\n"
        "        'full': 'meta_llama_llama_3_1_8b_instruct__full_8938_rows_full.csv',\n"
        "        'diag': 'meta_llama_llama_3_1_8b_instruct__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Llama 3.1 8B (Rerun)',\n"
        "        'family': 'standard',\n"
        "        'variant': 'rerun',\n"
        "        'size_b': 8,\n"
        "        'base_model': 'Llama 3.1 8B',\n"
        "        'preferred': True,\n"
        "        'notes': 'Rerun with updated standardized notebook',\n"
        "        'folder': 'Llama8B_ReRun',\n"
        "        'full': 'meta_llama_llama_3_1_8b_instruct__full_8938_rows_full_ReRun.csv',\n"
        "        'diag': 'meta_llama_llama_3_1_8b_instruct__full_8938_rows_diagnostics_ReRun.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Mistral Small 24B',\n"
        "        'family': 'standard',\n"
        "        'variant': 'current',\n"
        "        'size_b': 24,\n"
        "        'base_model': 'Mistral Small 24B',\n"
        "        'preferred': True,\n"
        "        'notes': 'Current 2501 Mistral run',\n"
        "        'folder': 'Mistral24B',\n"
        "        'full': 'mistralai_mistral_small_24b_instruct_2501__full_8938_rows_full.csv',\n"
        "        'diag': 'mistralai_mistral_small_24b_instruct_2501__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Qwen 2.5 14B (Original)',\n"
        "        'family': 'standard',\n"
        "        'variant': 'original',\n"
        "        'size_b': 14,\n"
        "        'base_model': 'Qwen 2.5 14B',\n"
        "        'preferred': False,\n"
        "        'notes': 'Original 14B run',\n"
        "        'folder': 'Qwen14B',\n"
        "        'full': 'per_model_total_output_qwen_qwen2_5_14b_instruct.csv',\n"
        "        'diag': 'qwen_qwen2_5_14b_instruct__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Qwen 2.5 14B (Rerun)',\n"
        "        'family': 'standard',\n"
        "        'variant': 'rerun',\n"
        "        'size_b': 14,\n"
        "        'base_model': 'Qwen 2.5 14B',\n"
        "        'preferred': True,\n"
        "        'notes': 'Rerun with updated standardized notebook',\n"
        "        'folder': 'QWEN14B_ReRun',\n"
        "        'full': 'qwen_qwen2_5_14b_instruct__full_8938_rows_full.csv',\n"
        "        'diag': 'qwen_qwen2_5_14b_instruct__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Qwen 2.5 32B (Original)',\n"
        "        'family': 'standard',\n"
        "        'variant': 'original',\n"
        "        'size_b': 32,\n"
        "        'base_model': 'Qwen 2.5 32B',\n"
        "        'preferred': False,\n"
        "        'notes': 'Original 32B run',\n"
        "        'folder': 'Qwen32B',\n"
        "        'full': 'per_model_total_output_qwen_qwen2_5_32b_instruct.csv',\n"
        "        'diag': 'qwen_qwen2_5_32b_instruct__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Qwen 2.5 32B (Rerun)',\n"
        "        'family': 'standard',\n"
        "        'variant': 'rerun',\n"
        "        'size_b': 32,\n"
        "        'base_model': 'Qwen 2.5 32B',\n"
        "        'preferred': True,\n"
        "        'notes': 'Rerun with updated standardized notebook',\n"
        "        'folder': 'QWEN32B_ReRun',\n"
        "        'full': 'qwen_qwen2_5_32b_instruct__full_8938_rows_full.csv',\n"
        "        'diag': 'qwen_qwen2_5_32b_instruct__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Qwen 2.5 72B',\n"
        "        'family': 'standard',\n"
        "        'variant': 'current',\n"
        "        'size_b': 72,\n"
        "        'base_model': 'Qwen 2.5 72B',\n"
        "        'preferred': True,\n"
        "        'notes': 'Current standard 72B run',\n"
        "        'folder': 'Qwen72B',\n"
        "        'full': 'qwen_qwen2_5_72b_instruct__full_8938_rows_full.csv',\n"
        "        'diag': 'qwen_qwen2_5_72b_instruct__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Llama 3.1 70B',\n"
        "        'family': 'standard',\n"
        "        'variant': 'current',\n"
        "        'size_b': 70,\n"
        "        'base_model': 'Llama 70B',\n"
        "        'preferred': True,\n"
        "        'notes': 'Current standard 70B run',\n"
        "        'folder': 'Llama70B',\n"
        "        'full': 'meta_llama_llama_3_1_70b_instruct__full_8938_rows_full.csv',\n"
        "        'diag': 'meta_llama_llama_3_1_70b_instruct__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Qwen 2.5 72B AWQ',\n"
        "        'family': 'quantized',\n"
        "        'variant': 'quantized',\n"
        "        'size_b': 72,\n"
        "        'base_model': 'Qwen 2.5 72B',\n"
        "        'preferred': True,\n"
        "        'notes': 'Quantized AWQ 72B run',\n"
        "        'folder': 'Qwen72B_quantized',\n"
        "        'full': 'qwen_2_5_72b_quantized__full_8938_rows_full.csv',\n"
        "        'diag': 'qwen_2_5_72b_quantized__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "    {\n"
        "        'label': 'Llama 3.3 70B AWQ',\n"
        "        'family': 'quantized',\n"
        "        'variant': 'quantized',\n"
        "        'size_b': 70,\n"
        "        'base_model': 'Llama 70B',\n"
        "        'preferred': True,\n"
        "        'notes': 'Quantized AWQ 70B run',\n"
        "        'folder': 'Llama70B_quantized',\n"
        "        'full': 'llama_3_3_70b_quantized__full_8938_rows_full.csv',\n"
        "        'diag': 'llama_3_3_70b_quantized__full_8938_rows_diagnostics.csv',\n"
        "    },\n"
        "]\n"
        "\n"
        "for spec in MODEL_SPECS:\n"
        "    spec['full_path'] = DATA_DIR / spec['folder'] / spec['full']\n"
        "    spec['diag_path'] = DATA_DIR / spec['folder'] / spec['diag']\n"
        "\n"
        "pd.DataFrame([{k: v for k, v in spec.items() if k in {'label', 'base_model', 'family', 'variant', 'preferred', 'folder'}} for spec in MODEL_SPECS])"
    )
)

cells.append(
    code_cell(
        "for spec in MODEL_SPECS:\n"
        "    assert spec['full_path'].exists(), f\"Missing full file: {spec['full_path']}\"\n"
        "    assert spec['diag_path'].exists(), f\"Missing diagnostics file: {spec['diag_path']}\"\n"
        "print('All configured model files found.')"
    )
)

cells.append(
    code_cell(
        "BG = '#0D0B14'\n"
        "PANEL = '#171221'\n"
        "GRID = '#3D315B'\n"
        "TEXT = '#F4EEFF'\n"
        "MUTED = '#CBBEFF'\n"
        "PINK = '#FF5D8F'\n"
        "VIOLET = '#8B6CFF'\n"
        "TEAL = '#00C2A8'\n"
        "GOLD = '#FFD166'\n"
        "SLATE = '#7E7599'\n"
        "\n"
        "DIM_COLORS = {\n"
        "    'informativeness': PINK,\n"
        "    'expressiveness': VIOLET,\n"
        "    'phatic': TEAL,\n"
        "    'creativeness_poeticness': GOLD,\n"
        "    'mixed': SLATE,\n"
        "}\n"
        "\n"
        "sns.set_theme(style='darkgrid')\n"
        "plt.rcParams.update({\n"
        "    'figure.facecolor': BG,\n"
        "    'axes.facecolor': PANEL,\n"
        "    'savefig.facecolor': BG,\n"
        "    'axes.edgecolor': GRID,\n"
        "    'axes.labelcolor': TEXT,\n"
        "    'xtick.color': TEXT,\n"
        "    'ytick.color': TEXT,\n"
        "    'text.color': TEXT,\n"
        "    'axes.titlecolor': TEXT,\n"
        "    'grid.color': GRID,\n"
        "    'font.size': 11,\n"
        "    'axes.titlesize': 16,\n"
        "    'axes.labelsize': 11,\n"
        "    'legend.frameon': True,\n"
        "    'legend.facecolor': PANEL,\n"
        "    'legend.edgecolor': GRID,\n"
        "})\n"
        "\n"
        "def nice_dim(x: str) -> str:\n"
        "    return x.replace('_', ' ').replace('creativeness poeticness', 'creativeness / poeticness').title()\n"
        "\n"
        "def style_ax(ax):\n"
        "    ax.set_facecolor(PANEL)\n"
        "    ax.grid(True, alpha=0.35)\n"
        "    for spine in ax.spines.values():\n"
        "        spine.set_color(GRID)\n"
        "    ax.tick_params(colors=TEXT)\n"
        "    ax.title.set_color(TEXT)\n"
        "    ax.xaxis.label.set_color(TEXT)\n"
        "    ax.yaxis.label.set_color(TEXT)\n"
        "\n"
        "def savefig(name: str):\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(FIGURES_DIR / name, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())\n"
        "\n"
        "PLOTLY_TEMPLATE = {\n"
        "    'layout': {\n"
        "        'paper_bgcolor': BG,\n"
        "        'plot_bgcolor': PANEL,\n"
        "        'font': {'color': TEXT, 'size': 14},\n"
        "        'xaxis': {'gridcolor': GRID, 'zerolinecolor': GRID, 'tickfont': {'color': TEXT}},\n"
        "        'yaxis': {'gridcolor': GRID, 'zerolinecolor': GRID, 'tickfont': {'color': TEXT}},\n"
        "        'legend': {'bgcolor': PANEL, 'bordercolor': GRID, 'font': {'color': TEXT}},\n"
        "        'title': {'font': {'color': TEXT, 'size': 26}},\n"
        "    }\n"
        "}\n"
    )
)

cells.append(
    code_cell(
        "def load_model_outputs(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:\n"
        "    full_df = pd.read_csv(spec['full_path'])\n"
        "    diag_df = pd.read_csv(spec['diag_path'])\n"
        "\n"
        "    if 'recovered_with_fallback' not in full_df.columns:\n"
        "        full_df['recovered_with_fallback'] = False\n"
        "\n"
        "    for col in SCORE_COLS:\n"
        "        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')\n"
        "\n"
        "    full_df['parse_ok'] = full_df['parse_ok'].fillna(False).astype(bool)\n"
        "    return full_df, diag_df\n"
        "\n"
        "frames = {}\n"
        "diag_frames = {}\n"
        "summary_rows = []\n"
        "\n"
        "for spec in MODEL_SPECS:\n"
        "    full_df, diag_df = load_model_outputs(spec)\n"
        "    frames[spec['label']] = full_df\n"
        "    diag_frames[spec['label']] = diag_df\n"
        "\n"
        "    ok_df = full_df[full_df['parse_ok']].copy()\n"
        "    row = {\n"
        "        'model_label': spec['label'],\n"
        "        'base_model': spec['base_model'],\n"
        "        'family': spec['family'],\n"
        "        'variant': spec['variant'],\n"
        "        'preferred': spec['preferred'],\n"
        "        'size_b': spec['size_b'],\n"
        "        'rows_total': len(full_df),\n"
        "        'rows_parse_ok': int(ok_df.shape[0]),\n"
        "        'rows_parse_fail': int((~full_df['parse_ok']).sum()),\n"
        "        'parse_ok_rate': float(diag_df.iloc[0].get('parse_ok_rate', ok_df.shape[0] / max(len(full_df), 1))),\n"
        "        'rows_per_second': float(diag_df.iloc[0].get('rows_per_second', np.nan)),\n"
        "        'total_seconds': float(diag_df.iloc[0].get('total_seconds', np.nan)),\n"
        "        'convergence_ratio': pd.to_numeric(diag_df.iloc[0].get('convergence_ratio', np.nan), errors='coerce'),\n"
        "        'last_chunk_mean_drift': pd.to_numeric(diag_df.iloc[0].get('last_chunk_mean_drift', np.nan), errors='coerce'),\n"
        "        'chunk_count': pd.to_numeric(diag_df.iloc[0].get('chunk_count', np.nan), errors='coerce'),\n"
        "        'convergence_proxy': diag_df.iloc[0].get('convergence_proxy', ''),\n"
        "        'fallback_recovered_rows': int(diag_df.iloc[0].get('fallback_recovered_rows', int(full_df['recovered_with_fallback'].sum()))),\n"
        "        'mean_confidence': float(ok_df['confidence'].mean()) if 'confidence' in ok_df.columns and len(ok_df) else np.nan,\n"
        "    }\n"
        "    for col in SCORE_COLS:\n"
        "        row[f'{col}_mean'] = float(ok_df[col].mean()) if len(ok_df) else np.nan\n"
        "        row[f'{col}_std'] = float(ok_df[col].std()) if len(ok_df) else np.nan\n"
        "    summary_rows.append(row)\n"
        "\n"
        "summary_df = pd.DataFrame(summary_rows).sort_values(['family', 'size_b']).reset_index(drop=True)\n"
        "summary_df['quality_speed_score'] = summary_df['parse_ok_rate'] * summary_df['rows_per_second']\n"
        "summary_df.to_csv(TABLES_DIR / 'summary_metrics.csv', index=False)\n"
        "summary_df"
    )
)

cells.append(
    code_cell(
        "# Row-aligned wide tables for each score dimension.\n"
        "score_wide = {}\n"
        "for col in SCORE_COLS:\n"
        "    pivot_parts = []\n"
        "    for label, df in frames.items():\n"
        "        tmp = df[['row_id', 'parse_ok', col]].copy()\n"
        "        tmp = tmp[tmp['parse_ok']].copy()\n"
        "        tmp['model_label'] = label\n"
        "        pivot_parts.append(tmp[['row_id', 'model_label', col]])\n"
        "    all_scores = pd.concat(pivot_parts, ignore_index=True)\n"
        "    score_wide[col] = all_scores.pivot(index='row_id', columns='model_label', values=col)\n"
        "\n"
        "dominant_parts = []\n"
        "for label, df in frames.items():\n"
        "    tmp = df.loc[df['parse_ok'], ['row_id', 'dominant_dimension']].copy()\n"
        "    tmp['model_label'] = label\n"
        "    dominant_parts.append(tmp)\n"
        "dominant_wide = pd.concat(dominant_parts, ignore_index=True).pivot(index='row_id', columns='model_label', values='dominant_dimension')\n"
        "\n"
        "list(score_wide.keys()), dominant_wide.shape"
    )
)

cells.append(
    code_cell(
        "def correlation_matrix(wide_df: pd.DataFrame) -> pd.DataFrame:\n"
        "    return wide_df.corr(method='pearson')\n"
        "\n"
        "def agreement_matrix(wide_df: pd.DataFrame, tolerance: float = 0.0) -> pd.DataFrame:\n"
        "    cols = list(wide_df.columns)\n"
        "    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)\n"
        "    for a in cols:\n"
        "        for b in cols:\n"
        "            if a == b:\n"
        "                mat.loc[a, b] = 1.0\n"
        "                continue\n"
        "            pair = wide_df[[a, b]].dropna()\n"
        "            if pair.empty:\n"
        "                mat.loc[a, b] = np.nan\n"
        "                continue\n"
        "            diff = (pair.iloc[:, 0] - pair.iloc[:, 1]).abs()\n"
        "            mat.loc[a, b] = float((diff <= tolerance).mean())\n"
        "    return mat\n"
        "\n"
        "def dominant_agreement_matrix(wide_df: pd.DataFrame) -> pd.DataFrame:\n"
        "    cols = list(wide_df.columns)\n"
        "    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)\n"
        "    for a in cols:\n"
        "        for b in cols:\n"
        "            if a == b:\n"
        "                mat.loc[a, b] = 1.0\n"
        "                continue\n"
        "            pair = wide_df[[a, b]].dropna()\n"
        "            mat.loc[a, b] = float((pair.iloc[:, 0] == pair.iloc[:, 1]).mean()) if len(pair) else np.nan\n"
        "    return mat\n"
        "\n"
        "def draw_heatmap(matrix: pd.DataFrame, title: str, filename: str, cmap: str, vmin=0, vmax=1):\n"
        "    fig, ax = plt.subplots(figsize=(8.6, 7.0))\n"
        "    sns.heatmap(\n"
        "        matrix,\n"
        "        ax=ax,\n"
        "        cmap=cmap,\n"
        "        vmin=vmin,\n"
        "        vmax=vmax,\n"
        "        annot=True,\n"
        "        fmt='.2f',\n"
        "        linewidths=0.5,\n"
        "        linecolor=GRID,\n"
        "        cbar_kws={'shrink': 0.82},\n"
        "    )\n"
        "    style_ax(ax)\n"
        "    ax.set_title(title)\n"
        "    ax.set_xlabel('')\n"
        "    ax.set_ylabel('')\n"
        "    plt.xticks(rotation=28, ha='right')\n"
        "    plt.yticks(rotation=0)\n"
        "    savefig(filename)\n"
        "    plt.show()\n"
    )
)

cells.append(
    md_cell(
        "## Summary tables\n"
        "\n"
        "This first block gives the main operational metrics used later in the figures."
    )
)

cells.append(
    code_cell(
        "display_cols = [\n"
        "    'model_label', 'base_model', 'family', 'variant', 'preferred', 'rows_total', 'rows_parse_ok', 'rows_parse_fail',\n"
        "    'parse_ok_rate', 'rows_per_second', 'total_seconds', 'convergence_ratio',\n"
        "    'last_chunk_mean_drift', 'mean_confidence', 'fallback_recovered_rows'\n"
        "]\n"
        "display(summary_df[display_cols].round(4))"
    )
)

cells.append(
    code_cell(
        "# Parse success, throughput, total runtime, convergence, and confidence.\n"
        "fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))\n"
        "tmp = summary_df.sort_values('parse_ok_rate', ascending=False).copy()\n"
        "\n"
        "sns.barplot(data=tmp, x='parse_ok_rate', y='model_label', hue='family', dodge=False, palette={'standard': VIOLET, 'quantized': TEAL}, ax=axes[0, 0])\n"
        "style_ax(axes[0, 0])\n"
        "axes[0, 0].set_title('Parse Success Rate by Model')\n"
        "axes[0, 0].set_xlabel('Parse success rate')\n"
        "axes[0, 0].set_ylabel('')\n"
        "\n"
        "tmp2 = summary_df.sort_values('rows_per_second', ascending=False).copy()\n"
        "sns.barplot(data=tmp2, x='rows_per_second', y='model_label', hue='family', dodge=False, palette={'standard': GOLD, 'quantized': PINK}, ax=axes[0, 1])\n"
        "style_ax(axes[0, 1])\n"
        "axes[0, 1].set_title('Throughput by Model')\n"
        "axes[0, 1].set_xlabel('Rows per second')\n"
        "axes[0, 1].set_ylabel('')\n"
        "\n"
        "sns.scatterplot(data=summary_df, x='last_chunk_mean_drift', y='mean_confidence', hue='family', style='family', s=160, palette={'standard': VIOLET, 'quantized': TEAL}, ax=axes[1, 0])\n"
        "style_ax(axes[1, 0])\n"
        "axes[1, 0].set_title('Confidence vs Late-Run Drift')\n"
        "axes[1, 0].set_xlabel('Late-run chunk drift')\n"
        "axes[1, 0].set_ylabel('Mean confidence')\n"
        "for _, row in summary_df.iterrows():\n"
        "    axes[1, 0].text(row['last_chunk_mean_drift'] + 0.003, row['mean_confidence'] + 0.003, row['model_label'], fontsize=8, color=TEXT)\n"
        "\n"
        "tmp3 = summary_df.sort_values('total_seconds', ascending=False).copy()\n"
        "sns.barplot(data=tmp3, x='total_seconds', y='model_label', hue='family', dodge=False, palette={'standard': '#F08BAA', 'quantized': '#7CE3D4'}, ax=axes[1, 1])\n"
        "style_ax(axes[1, 1])\n"
        "axes[1, 1].set_title('Total Runtime by Model')\n"
        "axes[1, 1].set_xlabel('Total seconds')\n"
        "axes[1, 1].set_ylabel('')\n"
        "\n"
        "for ax in axes.ravel():\n"
        "    legend = ax.get_legend()\n"
        "    if legend is not None:\n"
        "        legend.remove()\n"
        "\n"
        "savefig('01_runtime_quality_overview.png')\n"
        "plt.show()"
    )
)

cells.append(
    code_cell(
        "# Preferred-run-only operational view.\n"
        "preferred_df = summary_df[summary_df['preferred']].copy().sort_values(['family', 'size_b']).reset_index(drop=True)\n"
        "display(preferred_df[display_cols].round(4))\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))\n"
        "sns.barplot(data=preferred_df, x='model_label', y='parse_ok_rate', hue='family', dodge=False, palette={'standard': VIOLET, 'quantized': TEAL}, ax=axes[0])\n"
        "style_ax(axes[0])\n"
        "axes[0].set_title('Preferred Runs: Parse Success')\n"
        "axes[0].set_xlabel('')\n"
        "axes[0].set_ylabel('Parse success rate')\n"
        "axes[0].tick_params(axis='x', rotation=24)\n"
        "\n"
        "sns.barplot(data=preferred_df, x='model_label', y='rows_per_second', hue='family', dodge=False, palette={'standard': GOLD, 'quantized': PINK}, ax=axes[1])\n"
        "style_ax(axes[1])\n"
        "axes[1].set_title('Preferred Runs: Throughput')\n"
        "axes[1].set_xlabel('')\n"
        "axes[1].set_ylabel('Rows per second')\n"
        "axes[1].tick_params(axis='x', rotation=24)\n"
        "\n"
        "sns.barplot(data=preferred_df, x='model_label', y='last_chunk_mean_drift', hue='family', dodge=False, palette={'standard': '#F08BAA', 'quantized': '#7CE3D4'}, ax=axes[2])\n"
        "style_ax(axes[2])\n"
        "axes[2].set_title('Preferred Runs: Late-Run Drift')\n"
        "axes[2].set_xlabel('')\n"
        "axes[2].set_ylabel('Chunk drift')\n"
        "axes[2].tick_params(axis='x', rotation=24)\n"
        "\n"
        "for ax in axes:\n"
        "    leg = ax.get_legend()\n"
        "    if leg is not None:\n"
        "        leg.remove()\n"
        "savefig('01b_preferred_runtime_overview.png')\n"
        "plt.show()"
    )
)

cells.append(
    code_cell(
        "# Mean score profile across models.\n"
        "mean_long = summary_df.melt(\n"
        "    id_vars=['model_label', 'family', 'variant', 'base_model', 'preferred'],\n"
        "    value_vars=[f'{col}_mean' for col in SCORE_COLS],\n"
        "    var_name='dimension', value_name='mean_score'\n"
        ")\n"
        "mean_long['dimension'] = mean_long['dimension'].str.replace('_mean', '', regex=False)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(13.5, 6.2))\n"
        "sns.barplot(data=mean_long, x='model_label', y='mean_score', hue='dimension', palette=DIM_COLORS, ax=ax)\n"
        "style_ax(ax)\n"
        "ax.set_title('Mean Communication Scores by Model')\n"
        "ax.set_xlabel('')\n"
        "ax.set_ylabel('Mean score')\n"
        "plt.xticks(rotation=24, ha='right')\n"
        "handles, labels = ax.get_legend_handles_labels()\n"
        "ax.legend(handles, [nice_dim(x) for x in labels], ncol=2, title='Dimension')\n"
        "savefig('02_mean_scores_by_model.png')\n"
        "plt.show()"
    )
)

cells.append(
    code_cell(
        "# Mean score profile for preferred runs only.\n"
        "preferred_mean_long = mean_long[mean_long['preferred']].copy()\n"
        "fig, ax = plt.subplots(figsize=(13.5, 6.2))\n"
        "sns.barplot(data=preferred_mean_long, x='model_label', y='mean_score', hue='dimension', palette=DIM_COLORS, ax=ax)\n"
        "style_ax(ax)\n"
        "ax.set_title('Preferred Runs: Mean Communication Scores by Model')\n"
        "ax.set_xlabel('')\n"
        "ax.set_ylabel('Mean score')\n"
        "plt.xticks(rotation=24, ha='right')\n"
        "handles, labels = ax.get_legend_handles_labels()\n"
        "ax.legend(handles, [nice_dim(x) for x in labels], ncol=2, title='Dimension')\n"
        "savefig('02b_preferred_mean_scores.png')\n"
        "plt.show()"
    )
)

cells.append(
    code_cell(
        "# Standard deviation / variability profile.\n"
        "std_long = summary_df.melt(\n"
        "    id_vars=['model_label', 'family'],\n"
        "    value_vars=[f'{col}_std' for col in SCORE_COLS],\n"
        "    var_name='dimension', value_name='std_dev'\n"
        ")\n"
        "std_long['dimension'] = std_long['dimension'].str.replace('_std', '', regex=False)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(13.5, 6.2))\n"
        "sns.barplot(data=std_long, x='model_label', y='std_dev', hue='dimension', palette=DIM_COLORS, ax=ax)\n"
        "style_ax(ax)\n"
        "ax.set_title('Score Variability by Model and Dimension')\n"
        "ax.set_xlabel('')\n"
        "ax.set_ylabel('Standard deviation')\n"
        "plt.xticks(rotation=24, ha='right')\n"
        "handles, labels = ax.get_legend_handles_labels()\n"
        "ax.legend(handles, [nice_dim(x) for x in labels], ncol=2, title='Dimension')\n"
        "savefig('03_dimension_variability.png')\n"
        "plt.show()"
    )
)

cells.append(
    code_cell(
        "# Dominant-dimension shares.\n"
        "dominant_rows = []\n"
        "for label, df in frames.items():\n"
        "    ok = df[df['parse_ok']].copy()\n"
        "    shares = ok['dominant_dimension'].value_counts(normalize=True)\n"
        "    spec = next(spec for spec in MODEL_SPECS if spec['label'] == label)\n"
        "    row = {'model_label': label, 'base_model': spec['base_model'], 'family': spec['family'], 'variant': spec['variant'], 'preferred': spec['preferred']}\n"
        "    for dim in DOM_ORDER:\n"
        "        row[dim] = float(shares.get(dim, 0.0))\n"
        "    dominant_rows.append(row)\n"
        "dominant_df = pd.DataFrame(dominant_rows)\n"
        "dominant_df.to_csv(TABLES_DIR / 'dominant_dimension_shares.csv', index=False)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(13.5, 6.5))\n"
        "style_ax(ax)\n"
        "bottom = np.zeros(len(dominant_df))\n"
        "for dim in DOM_ORDER:\n"
        "    vals = dominant_df[dim].values\n"
        "    ax.bar(dominant_df['model_label'], vals, bottom=bottom, color=DIM_COLORS[dim], label=nice_dim(dim))\n"
        "    bottom += vals\n"
        "ax.set_title('Dominant Communication Dimension by Model')\n"
        "ax.set_ylabel('Share of parsed rows')\n"
        "ax.set_xlabel('')\n"
        "ax.set_ylim(0, 1)\n"
        "plt.xticks(rotation=24, ha='right')\n"
        "ax.legend(ncol=3, title='Dominant dimension')\n"
        "savefig('04_dominant_dimension_shares.png')\n"
        "plt.show()\n"
        "\n"
        "dominant_df"
    )
)

cells.append(
    code_cell(
        "# Preferred-run dominant shares only.\n"
        "preferred_dom = dominant_df[dominant_df['preferred']].copy().reset_index(drop=True)\n"
        "fig, ax = plt.subplots(figsize=(13.5, 6.5))\n"
        "style_ax(ax)\n"
        "bottom = np.zeros(len(preferred_dom))\n"
        "for dim in DOM_ORDER:\n"
        "    vals = preferred_dom[dim].values\n"
        "    ax.bar(preferred_dom['model_label'], vals, bottom=bottom, color=DIM_COLORS[dim], label=nice_dim(dim))\n"
        "    bottom += vals\n"
        "ax.set_title('Preferred Runs: Dominant Dimension Profile by Model')\n"
        "ax.set_ylabel('Share of parsed rows')\n"
        "ax.set_xlabel('')\n"
        "ax.set_ylim(0, 1)\n"
        "plt.xticks(rotation=24, ha='right')\n"
        "ax.legend(ncol=3, title='Dominant dimension')\n"
        "savefig('04b_preferred_dominant_dimension_shares.png')\n"
        "plt.show()"
    )
)

cells.append(
    code_cell(
        "# Standard vs quantized comparison for the two large-model families.\n"
        "pair_df = pd.DataFrame([\n"
        "    {'family_pair': 'Llama 70B', 'variant': 'standard', 'rows_per_second': summary_df.loc[summary_df['model_label'] == 'Llama 3.1 70B', 'rows_per_second'].iloc[0], 'parse_ok_rate': summary_df.loc[summary_df['model_label'] == 'Llama 3.1 70B', 'parse_ok_rate'].iloc[0]},\n"
        "    {'family_pair': 'Llama 70B', 'variant': 'quantized', 'rows_per_second': summary_df.loc[summary_df['model_label'] == 'Llama 3.3 70B AWQ', 'rows_per_second'].iloc[0], 'parse_ok_rate': summary_df.loc[summary_df['model_label'] == 'Llama 3.3 70B AWQ', 'parse_ok_rate'].iloc[0]},\n"
        "    {'family_pair': 'Qwen 72B', 'variant': 'standard', 'rows_per_second': summary_df.loc[summary_df['model_label'] == 'Qwen 2.5 72B', 'rows_per_second'].iloc[0], 'parse_ok_rate': summary_df.loc[summary_df['model_label'] == 'Qwen 2.5 72B', 'parse_ok_rate'].iloc[0]},\n"
        "    {'family_pair': 'Qwen 72B', 'variant': 'quantized', 'rows_per_second': summary_df.loc[summary_df['model_label'] == 'Qwen 2.5 72B AWQ', 'rows_per_second'].iloc[0], 'parse_ok_rate': summary_df.loc[summary_df['model_label'] == 'Qwen 2.5 72B AWQ', 'parse_ok_rate'].iloc[0]},\n"
        "])\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))\n"
        "sns.barplot(data=pair_df, x='family_pair', y='rows_per_second', hue='variant', palette={'standard': VIOLET, 'quantized': TEAL}, ax=axes[0])\n"
        "style_ax(axes[0])\n"
        "axes[0].set_title('Large Models: Throughput Gain from Quantization')\n"
        "axes[0].set_xlabel('')\n"
        "axes[0].set_ylabel('Rows per second')\n"
        "\n"
        "sns.barplot(data=pair_df, x='family_pair', y='parse_ok_rate', hue='variant', palette={'standard': PINK, 'quantized': GOLD}, ax=axes[1])\n"
        "style_ax(axes[1])\n"
        "axes[1].set_title('Large Models: Parse Success After Quantization')\n"
        "axes[1].set_xlabel('')\n"
        "axes[1].set_ylabel('Parse success rate')\n"
        "\n"
        "for ax in axes:\n"
        "    ax.legend(title='Variant')\n"
        "\n"
        "savefig('05_quantized_vs_standard_large_models.png')\n"
        "plt.show()\n"
        "\n"
        "pair_df"
    )
)

cells.append(
    code_cell(
        "# Version-to-version comparison for the models with multiple stored runs.\n"
        "version_compare = summary_df[summary_df['base_model'].isin(['Gemma 4 26B', 'Llama 3.1 8B', 'Qwen 2.5 14B', 'Qwen 2.5 32B'])].copy()\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))\n"
        "sns.barplot(data=version_compare, x='base_model', y='parse_ok_rate', hue='variant', ax=axes[0], palette='magma')\n"
        "style_ax(axes[0])\n"
        "axes[0].set_title('Original vs Rerun / Regex: Parse Success')\n"
        "axes[0].set_xlabel('')\n"
        "axes[0].set_ylabel('Parse success rate')\n"
        "axes[0].tick_params(axis='x', rotation=18)\n"
        "\n"
        "sns.barplot(data=version_compare, x='base_model', y='rows_per_second', hue='variant', ax=axes[1], palette='viridis')\n"
        "style_ax(axes[1])\n"
        "axes[1].set_title('Original vs Rerun / Regex: Throughput')\n"
        "axes[1].set_xlabel('')\n"
        "axes[1].set_ylabel('Rows per second')\n"
        "axes[1].tick_params(axis='x', rotation=18)\n"
        "savefig('05b_version_comparison.png')\n"
        "plt.show()\n"
        "\n"
        "version_compare[['model_label', 'base_model', 'variant', 'parse_ok_rate', 'rows_per_second', 'last_chunk_mean_drift']].sort_values(['base_model', 'variant'])"
    )
)

cells.append(md_cell("## Correlation matrices by dimension"))

cells.append(
    code_cell(
        "correlation_tables = {}\n"
        "for idx, dim in enumerate(SCORE_COLS, start=1):\n"
        "    corr = correlation_matrix(score_wide[dim])\n"
        "    correlation_tables[dim] = corr\n"
        "    corr.to_csv(TABLES_DIR / f'{dim}_correlation_matrix.csv')\n"
        "    draw_heatmap(corr, f'{nice_dim(dim)} Correlation Across Models', f'06_{idx}_{dim}_correlation_matrix.png', cmap='mako')\n"
        "\n"
        "correlation_tables['informativeness']"
    )
)

cells.append(
    code_cell(
        "# Preferred-run-only correlation matrices.\n"
        "preferred_labels = preferred_df['model_label'].tolist()\n"
        "preferred_corr_tables = {}\n"
        "for idx, dim in enumerate(SCORE_COLS, start=1):\n"
        "    corr = correlation_matrix(score_wide[dim][preferred_labels])\n"
        "    preferred_corr_tables[dim] = corr\n"
        "    corr.to_csv(TABLES_DIR / f'preferred_{dim}_correlation_matrix.csv')\n"
        "    draw_heatmap(corr, f'Preferred Runs: {nice_dim(dim)} Correlation', f'06b_{idx}_{dim}_preferred_correlation_matrix.png', cmap='mako')\n"
        "preferred_corr_tables['informativeness']"
    )
)

cells.append(md_cell("## Agreement matrices by dimension"))

cells.append(
    code_cell(
        "exact_tables = {}\n"
        "within1_tables = {}\n"
        "for idx, dim in enumerate(SCORE_COLS, start=1):\n"
        "    exact = agreement_matrix(score_wide[dim], tolerance=0.0)\n"
        "    within1 = agreement_matrix(score_wide[dim], tolerance=1.0)\n"
        "    exact_tables[dim] = exact\n"
        "    within1_tables[dim] = within1\n"
        "    exact.to_csv(TABLES_DIR / f'{dim}_exact_agreement.csv')\n"
        "    within1.to_csv(TABLES_DIR / f'{dim}_within1_agreement.csv')\n"
        "    draw_heatmap(exact, f'{nice_dim(dim)} Exact Agreement', f'07_{idx}_{dim}_exact_agreement.png', cmap='rocket')\n"
        "    draw_heatmap(within1, f'{nice_dim(dim)} Within-1 Agreement', f'08_{idx}_{dim}_within1_agreement.png', cmap='crest')\n"
        "\n"
        "within1_tables['expressiveness']"
    )
)

cells.append(
    code_cell(
        "# Preferred-run-only within-1 agreement.\n"
        "preferred_within1_tables = {}\n"
        "for idx, dim in enumerate(SCORE_COLS, start=1):\n"
        "    within1 = agreement_matrix(score_wide[dim][preferred_labels], tolerance=1.0)\n"
        "    preferred_within1_tables[dim] = within1\n"
        "    within1.to_csv(TABLES_DIR / f'preferred_{dim}_within1_agreement.csv')\n"
        "    draw_heatmap(within1, f'Preferred Runs: {nice_dim(dim)} Within-1 Agreement', f'08b_{idx}_{dim}_preferred_within1_agreement.png', cmap='crest')\n"
        "preferred_within1_tables['expressiveness']"
    )
)

cells.append(
    code_cell(
        "# Dominant-dimension agreement matrix.\n"
        "dom_agree = dominant_agreement_matrix(dominant_wide)\n"
        "dom_agree.to_csv(TABLES_DIR / 'dominant_dimension_agreement.csv')\n"
        "draw_heatmap(dom_agree, 'Dominant-Dimension Agreement Across Models', '09_dominant_dimension_agreement.png', cmap='flare')\n"
        "dom_agree"
    )
)

cells.append(
    code_cell(
        "# Aggregate cross-model agreement summary.\n"
        "avg_within1 = pd.DataFrame({dim: within1_tables[dim].mean(axis=1) for dim in SCORE_COLS})\n"
        "avg_within1['overall_avg'] = avg_within1.mean(axis=1)\n"
        "avg_within1 = avg_within1.reset_index().rename(columns={'index': 'model_label'})\n"
        "avg_within1.to_csv(TABLES_DIR / 'average_within1_by_model.csv', index=False)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(13.5, 5.8))\n"
        "plot_df = avg_within1.melt(id_vars='model_label', value_vars=SCORE_COLS, var_name='dimension', value_name='avg_within1')\n"
        "sns.barplot(data=plot_df, x='model_label', y='avg_within1', hue='dimension', palette=DIM_COLORS, ax=ax)\n"
        "style_ax(ax)\n"
        "ax.set_title('Average Within-1 Agreement by Model and Dimension')\n"
        "ax.set_xlabel('')\n"
        "ax.set_ylabel('Average within-1 agreement')\n"
        "plt.xticks(rotation=24, ha='right')\n"
        "handles, labels = ax.get_legend_handles_labels()\n"
        "ax.legend(handles, [nice_dim(x) for x in labels], ncol=2, title='Dimension')\n"
        "savefig('10_average_within1_by_model.png')\n"
        "plt.show()\n"
        "\n"
        "avg_within1"
    )
)

cells.append(
    code_cell(
        "# Preferred-run agreement summary.\n"
        "preferred_avg_within1 = pd.DataFrame({dim: preferred_within1_tables[dim].mean(axis=1) for dim in SCORE_COLS})\n"
        "preferred_avg_within1['overall_avg'] = preferred_avg_within1.mean(axis=1)\n"
        "preferred_avg_within1 = preferred_avg_within1.reset_index().rename(columns={'index': 'model_label'})\n"
        "preferred_avg_within1.to_csv(TABLES_DIR / 'preferred_average_within1_by_model.csv', index=False)\n"
        "preferred_avg_within1.sort_values('overall_avg', ascending=False)"
    )
)

cells.append(
    code_cell(
        "# Pairwise mean absolute difference summary.\n"
        "pair_rows = []\n"
        "labels = list(frames.keys())\n"
        "for i, a in enumerate(labels):\n"
        "    for b in labels[i+1:]:\n"
        "        for dim in SCORE_COLS:\n"
        "            pair = score_wide[dim][[a, b]].dropna()\n"
            "            mad = (pair[a] - pair[b]).abs().mean() if len(pair) else np.nan\n"
            "            pair_rows.append({'model_a': a, 'model_b': b, 'dimension': dim, 'mean_abs_difference': mad, 'overlap': len(pair)})\n"
        "pairwise_diff_df = pd.DataFrame(pair_rows)\n"
        "pairwise_diff_df.to_csv(TABLES_DIR / 'pairwise_mean_abs_difference.csv', index=False)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(14, 6.2))\n"
        "plot_df = pairwise_diff_df.copy()\n"
        "plot_df['pair'] = plot_df['model_a'] + ' vs ' + plot_df['model_b']\n"
        "sns.barplot(data=plot_df, x='pair', y='mean_abs_difference', hue='dimension', palette=DIM_COLORS, ax=ax)\n"
        "style_ax(ax)\n"
        "ax.set_title('Pairwise Mean Absolute Score Differences')\n"
        "ax.set_xlabel('')\n"
        "ax.set_ylabel('Mean absolute difference')\n"
        "plt.xticks(rotation=40, ha='right')\n"
        "handles, labels = ax.get_legend_handles_labels()\n"
        "ax.legend(handles, [nice_dim(x) for x in labels], ncol=2, title='Dimension')\n"
        "savefig('11_pairwise_mean_abs_difference.png')\n"
        "plt.show()\n"
        "\n"
        "pairwise_diff_df.sort_values('mean_abs_difference', ascending=False).head(12)"
    )
)

cells.append(
    code_cell(
        "# Interactive version of the pairwise difference chart for zooming and scrolling.\n"
        "plot_df = pairwise_diff_df.copy()\n"
        "plot_df['pair'] = plot_df['model_a'] + ' vs ' + plot_df['model_b']\n"
        "color_map = {\n"
        "    'informativeness': PINK,\n"
        "    'expressiveness': VIOLET,\n"
        "    'phatic': TEAL,\n"
        "    'creativeness_poeticness': GOLD,\n"
        "}\n"
        "fig = px.bar(\n"
        "    plot_df,\n"
        "    x='pair',\n"
        "    y='mean_abs_difference',\n"
        "    color='dimension',\n"
        "    barmode='group',\n"
        "    color_discrete_map=color_map,\n"
        "    hover_data=['model_a', 'model_b', 'dimension', 'mean_abs_difference', 'overlap'],\n"
        "    title='Interactive Pairwise Mean Absolute Score Differences',\n"
        "    height=900,\n"
        "    width=max(2200, 18 * plot_df['pair'].nunique()),\n"
        ")\n"
        "fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title='', yaxis_title='Mean absolute difference')\n"
        "fig.update_xaxes(tickangle=-45)\n"
        "fig.show()\n"
        "\n"
        "# Save an HTML copy so the interactive chart can be reopened later in a browser.\n"
        "fig.write_html(FIGURES_DIR / '11b_pairwise_mean_abs_difference_interactive.html', include_plotlyjs='cdn')\n"
        "print(FIGURES_DIR / '11b_pairwise_mean_abs_difference_interactive.html')"
    )
)

cells.append(
    code_cell(
        "# Interactive version of the runtime-quality scatter for easier label inspection.\n"
        "scatter_df = summary_df.copy()\n"
        "fig = px.scatter(\n"
        "    scatter_df,\n"
        "    x='last_chunk_mean_drift',\n"
        "    y='mean_confidence',\n"
        "    color='family',\n"
        "    symbol='variant',\n"
        "    text='model_label',\n"
        "    hover_data=['model_label', 'base_model', 'variant', 'parse_ok_rate', 'rows_per_second', 'total_seconds'],\n"
        "    color_discrete_map={'standard': VIOLET, 'quantized': TEAL},\n"
        "    title='Interactive Confidence vs Late-Run Drift',\n"
        "    height=700,\n"
        "    width=1100,\n"
        ")\n"
        "fig.update_traces(textposition='top center', marker=dict(size=14, line=dict(width=1, color=TEXT)))\n"
        "fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title='Late-run chunk drift', yaxis_title='Mean confidence')\n"
        "fig.show()\n"
        "fig.write_html(FIGURES_DIR / '01c_confidence_vs_drift_interactive.html', include_plotlyjs='cdn')\n"
        "print(FIGURES_DIR / '01c_confidence_vs_drift_interactive.html')"
    )
)

cells.append(
    code_cell(
        "# GPU memory footprint estimate parsed from diagnostics strings.\n"
        "import ast\n"
        "\n"
        "def parse_peak_used_gb(raw):\n"
        "    try:\n"
        "        items = ast.literal_eval(raw)\n"
        "        return max(float(x.get('used_gb', np.nan)) for x in items)\n"
        "    except Exception:\n"
        "        return np.nan\n"
        "\n"
        "gpu_rows = []\n"
        "for label, diag in diag_frames.items():\n"
        "    row = diag.iloc[0]\n"
        "    gpu_rows.append({\n"
        "        'model_label': label,\n"
        "        'peak_used_gb': parse_peak_used_gb(row.get('gpu_after_load', '')),\n"
        "        'rows_per_second': float(row.get('rows_per_second', np.nan)),\n"
        "        'parse_ok_rate': float(row.get('parse_ok_rate', np.nan)),\n"
        "    })\n"
        "gpu_df = pd.DataFrame(gpu_rows)\n"
        "gpu_df.to_csv(TABLES_DIR / 'gpu_memory_summary.csv', index=False)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(13.5, 5.4))\n"
        "sns.scatterplot(data=gpu_df, x='peak_used_gb', y='rows_per_second', hue='parse_ok_rate', palette='viridis', s=170, ax=ax)\n"
        "style_ax(ax)\n"
        "ax.set_title('Peak GPU Memory vs Throughput')\n"
        "ax.set_xlabel('Peak used GPU memory per device (GB)')\n"
        "ax.set_ylabel('Rows per second')\n"
        "for _, row in gpu_df.iterrows():\n"
        "    ax.text(row['peak_used_gb'] + 0.2, row['rows_per_second'] + 0.05, row['model_label'], fontsize=8, color=TEXT)\n"
        "savefig('12_peak_gpu_memory_vs_throughput.png')\n"
        "plt.show()\n"
        "\n"
        "gpu_df.sort_values('peak_used_gb', ascending=False)"
    )
)

cells.append(
    code_cell(
        "# Export a compact cross-model report table.\n"
        "report_table = summary_df[['model_label', 'base_model', 'family', 'variant', 'preferred', 'parse_ok_rate', 'rows_per_second', 'total_seconds', 'convergence_ratio', 'last_chunk_mean_drift', 'mean_confidence']].copy()\n"
        "for dim in SCORE_COLS:\n"
        "    report_table[f'{dim}_mean'] = summary_df[f'{dim}_mean']\n"
        "report_table.to_csv(TABLES_DIR / 'final_model_report_table.csv', index=False)\n"
        "report_table.round(4)"
    )
)

cells.append(
    md_cell(
        "## Notes\n"
        "\n"
        "- This notebook includes all current run folders in the repository, including original runs, reruns, regex-enhanced runs, and quantized runs.\n"
        "- Preferred runs are marked in the summary table and also receive their own dedicated figures.\n"
        "- Both standard and quantized 70B / 72B variants are included in the same comparison.\n"
        "- All summary tables and figures are exported to:\n"
        "\n"
        "  - `/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/analysis_notebooks/model_comparison_outputs/tables`\n"
        "  - `/Users/raresolteanu/Desktop/Gliner-Work.Dauphine/analysis_notebooks/model_comparison_outputs/figures`\n"
    )
)


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote notebook to {NOTEBOOK_PATH}")
