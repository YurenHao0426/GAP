# GAP — Generalization-and-Perturbation Framework

[![arXiv](https://img.shields.io/badge/arXiv-2508.08833-b31b1b.svg)](https://arxiv.org/abs/2508.08833)
[![Hugging Face](https://img.shields.io/badge/🤗_Dataset-PutnamGAP-yellow.svg)](https://huggingface.co/datasets/blackhao0426/PutnamGAP)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**GAP** (*Generalization-and-Perturbation*) is an automatable evaluation framework for stress-testing the **robustness of LLM mathematical reasoning** under semantically equivalent transformations of advanced math problems. It partitions equivalence-preserving transformations into two qualitatively different families — **surface renaming** and **kernel parameter resampling** — and provides paired-evaluation, mechanism-sensitive analyses that prior perturbation benchmarks cannot.

This repository contains the **complete pipeline source code**:

- The variant generation and evaluation pipeline used to build PutnamGAP and to evaluate 18 LLMs
- The label-free structural-overlap analysis used in our framework-level mechanism analyses
- The repairability rescue harness with three-condition prefix injection
- Auxiliary cleaning, audit, aggregation, and figure-rendering scripts

The companion **PutnamGAP dataset** (1,051 Putnam problems × 5 mathematically equivalent variants = 6,306 items) is hosted on Hugging Face: <https://huggingface.co/datasets/blackhao0426/PutnamGAP>.

The accompanying paper is on arXiv: [2508.08833](https://arxiv.org/abs/2508.08833).

---

## What this repo provides

| Directory | Contents |
|---|---|
| `putnam-bench-anon/` | The main GAP CLI and pipeline. Loaders for OpenAI / Anthropic / Google / xAI / OpenRouter / vLLM, multi-judge variant verification (`scripts/`), surface and kernel variant generators, end-to-end evaluation runner (`putnam_cli.py`), setup helper, install script, and per-provider prompt templates. |
| `putnamsup/` | Standalone runners using the OpenRouter API and a local-model HuggingFace inference path. Includes `evaluate_putnam_gap.py`, `run_putnam_gap.py`, `run_putnam_gap_openrouter.py`, and the `putnamgap_viewer.py` browser. |
| `analysis/` | Framework-level mechanism analyses: paired structural overlap (`structural_overlap.py`, `kv_overlap.py`, `aggregate_overlap.py`), repairability rescue (`rescue_runner.py`, `rescue_prompts.py`, `rescue_api.py`, `rescue_analyze.py`, `rescue_pooled.py`), self-correction probe (`self_correction.py`, `sc_success_and_difficulty.py`), cross-model agreement (`cross_model_agreement.py`), topic × problem-type interaction (`topic_problemtype_interaction.py`), spontaneous-normalization sub-finding (`normalization_analysis.py`), figure rendering (`make_figures.py`), Unicode → LaTeX cleaner and audit (`unicode_clean.py`, `unicode_audit.py`, `balance_diff.py`, `spotcheck_clean.py`). |
| `mini_gap_math*.py`, `kv_math*.py` | Stand-alone scripts used to instantiate GAP on the MATH benchmark (Mini-GAP-MATH) and to run kernel-variant generation experiments. |

---

## Installation

```bash
git clone https://github.com/YurenHao0426/GAP.git
cd GAP/putnam-bench-anon
python -m pip install -r requirements.txt   # or requirements-minimal.txt for CPU-only
```

Set the provider API keys you intend to use as environment variables (the pipeline reads them via `os.getenv` — there are no hard-coded credentials):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...           # optional
export OPENROUTER_API_KEY=...    # optional
```

Then download the PutnamGAP dataset:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('blackhao0426/PutnamGAP', repo_type='dataset',
                  local_dir='./putnam-bench-anon/dataset', allow_patterns='dataset/*.json')
"
```

(If you prefer, you can also use `datasets.load_dataset('blackhao0426/PutnamGAP')` directly inside Python and bypass the on-disk layout.)

---

## Quick start

### 1. Run the GAP evaluation pipeline on PutnamGAP

```bash
cd putnam-bench-anon
python putnam_cli.py --solver-model gpt-4o-mini --grader-model gpt-4o \
  --variant original --output ../runs/gpt-4o-mini-original.json
python putnam_cli.py --solver-model gpt-4o-mini --grader-model gpt-4o \
  --variant garbled_string --output ../runs/gpt-4o-mini-gs.json
```

The CLI iterates over the 1,051 problems in the configured `dataset/` directory, calls the solver model, scores the response with the grader model, and writes a structured JSON results file containing per-problem solve and grade records (`solve.solution`, `grade.grade`, `correct`, etc.).

### 2. Generate framework-level mechanism analyses

After you have run the solver pipeline on at least the original variant and one surface variant, you can run the label-free structural overlap analysis:

```bash
python analysis/structural_overlap.py
python analysis/aggregate_overlap.py
```

This will produce per-cell Cohen's *d* statistics for the stable-vs-brittle structural overlap dichotomy described in the paper.

### 3. Run the repairability rescue experiment

```bash
python analysis/rescue_runner.py --pilot       # 5 cases per cell smoke test
python analysis/rescue_runner.py --cases 30    # full rescue run
python analysis/rescue_pooled.py               # pooled summary tables
```

The rescue harness re-solves each flip case under three prefix conditions (`canonical_T2`, `own_T2`, `null`) using the same model under test, then grades the new attempt.

### 4. Plot the headline figures

```bash
python analysis/make_figures.py
```

---

## Reproducing the published results

The exact configuration we used is:

| Step | Models |
|---|---|
| Solver evaluation (18 models) | OpenAI o3, o4-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini; Anthropic claude-opus-4, claude-sonnet-4; Google gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite; xAI grok-4; Alibaba qwen3-235B; Meta llama-4 maverick; Moonshot kimi-k2; DeepSeek deepseek-prover; Mistral devstral-medium |
| Kernel variant generation (5-judge unanimous) | o3, claude-sonnet-4, gemini-2.5-flash, gpt-4.1-mini, gpt-4o |
| Grader (paper) | o3 (primary) with three-provider cross-check via gpt-4o + claude-sonnet-4 + gemini-2.5-flash on a stratified subset (κ = 0.96–1.00) |
| Rescue experiment (this repo) | 4 representative models: gpt-4.1-mini, gpt-4o-mini, claude-sonnet-4, gemini-2.5-flash, with gpt-4o as the grader |

The three-provider grader cross-check protocol is documented in `putnam-bench-anon/scripts/regrade.py`. The 5-judge kernel variant verification is documented in `putnam-bench-anon/scripts/compare_original_vs_kernel_test.py`.

---

## Important: Source Attribution

> **The original Putnam Competition problem statements and the canonical solutions in PutnamGAP are reproduced from four authoritative monographs published by the Mathematical Association of America (MAA Press), under the fair-use clause printed in the front-matter of every volume:**
>
> *"Individual readers ... are permitted to make fair use of the material, such as to copy select pages for use in teaching or research."*
>
> **All original problem statements and canonical solutions remain the intellectual property of the MAA. If you use this code or the PutnamGAP dataset for any research output, you MUST cite both the GAP framework paper AND the four MAA source books listed below. Failure to do so misrepresents the provenance of the original problems.**

Problem and solution sets from 2017 onward are included in PutnamGAP with the explicit permission of MAA.

**Takedown notice.** If you are an author, publisher, or rights-holder and you believe any portion of this release infringes your rights, please open an issue at <https://github.com/YurenHao0426/GAP/issues> or email the maintainer. The affected items will be removed promptly.

---

## Citation

If you use this code or the PutnamGAP dataset, you **must** cite **all five** entries below: the GAP framework paper **and** the four MAA Putnam source books that the original problems and solutions are reproduced from. Citing fewer is a misrepresentation of the dataset's provenance.

In-text example:

> "We evaluate on PutnamGAP \cite{hao2025gap, putnamI, putnamII, putnamIII, putnamIV}."

Full BibTeX (copy the entire block — all five entries are mandatory):

```bibtex
@article{hao2025gap,
  title   = {An Investigation of Robustness of {LLM}s in Mathematical Reasoning:
             Benchmarking with Mathematically-Equivalent Transformation of
             Advanced Mathematical Problems},
  author  = {Hao, Yuren and Wan, Xiang and Zhai, ChengXiang},
  journal = {arXiv preprint arXiv:2508.08833},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.08833}
}

@book{putnamI,
  title     = {The William Lowell Putnam Mathematical Competition:
               Problems and Solutions 1938--1964},
  author    = {Gleason, A. M. and Greenwood, R. E. and Kelly, L. M.},
  publisher = {Mathematical Association of America},
  year      = {1980},
  series    = {MAA Problem Books},
  volume    = {1},
  address   = {Washington, DC},
  note      = {673\,pp; reprinted by AMS/MAA Press}
}

@book{putnamII,
  title     = {The William Lowell Putnam Mathematical Competition:
               Problems and Solutions 1965--1984},
  author    = {Alexanderson, Gerald L. and Klosinski, Leonard F. and
               Larson, Loren C.},
  publisher = {Mathematical Association of America},
  year      = {1985},
  series    = {MAA Problem Books},
  volume    = {30},
  address   = {Washington, DC},
  note      = {Reprinted by AMS/MAA Press}
}

@book{putnamIII,
  title     = {The William Lowell Putnam Mathematical Competition 1985--2000:
               Problems, Solutions and Commentary},
  author    = {Kedlaya, Kiran S. and Poonen, Bjorn and Vakil, Ravi},
  publisher = {Mathematical Association of America},
  year      = {2002},
  series    = {MAA Problem Books},
  volume    = {33},
  address   = {Washington, DC},
  note      = {Reprinted by AMS/MAA Press}
}

@book{putnamIV,
  title     = {The William Lowell Putnam Mathematical Competition 2001--2016:
               Problems, Solutions and Commentary},
  author    = {Kedlaya, Kiran S. and Kane, Daniel M. and Kane, Jonathan M. and
               O'Dorney, Evan M.},
  publisher = {American Mathematical Society (MAA Press)},
  year      = {2020},
  series    = {MAA Problem Books},
  volume    = {37},
  address   = {Providence, RI},
  note      = {Softcover and e-book versions available}
}
```

> **Reminder.** The four `putnamI`–`putnamIV` entries are not optional or supplementary; the original problem statements and canonical solutions in PutnamGAP are reproduced from those four MAA monographs under the MAA fair-use clause, and the IP belongs to the Mathematical Association of America. Any downstream use of this code or dataset that omits the four MAA citations misrepresents the dataset's provenance.

---

## License

- The **pipeline source code**, **variant generation scripts**, **evaluation harness**, **structural-overlap analysis**, **rescue runner**, **cleaning/audit tools**, and any other artefact authored by the GAP project is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- The **original Putnam Competition problem statements and canonical solutions** that the pipeline operates on remain copyrighted by the Mathematical Association of America (MAA). They are redistributed in the companion PutnamGAP dataset under MAA's stated fair-use clause and only for educational and research use. **Downstream users must cite the four MAA source books listed above.**

---

## Links

- **Paper (arXiv)**: <https://arxiv.org/abs/2508.08833>
- **GAP framework code (this repo)**: <https://github.com/YurenHao0426/GAP>
- **PutnamGAP dataset (Hugging Face — primary)**: <https://huggingface.co/datasets/blackhao0426/PutnamGAP>
- **PutnamGAP dataset (GitHub mirror)**: <https://github.com/YurenHao0426/PutnamGAP>
- **Issues & contact**: <https://github.com/YurenHao0426/GAP/issues>
