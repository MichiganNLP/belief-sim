# BeliefSim

**BeliefSim: Towards Belief-Driven Simulation of Demographic Misinformation Susceptibility**

BeliefSim is a research codebase for simulating misinformation susceptibility with large language models using belief evidence rather than demographic personas alone. The repository contains scripts for prompt conditioning, WVS belief-prior construction, PANDORA/MIST evaluation, counterfactual demographic tests, fine-tuning, analysis, dataset release, and the project website.

- Paper: https://arxiv.org/abs/2603.03585
- Project website: https://michigannlp.github.io/belief-sim/
- Code: https://github.com/MichiganNLP/belief-sim
- Dataset: https://huggingface.co/datasets/Angana192/beliefsim

## What This Code Does

BeliefSim represents each evaluation instance with:

1. **Target claim**: the held-out misinformation/news claim whose participant judgment is predicted.
2. **Observed beliefs**: two other same-participant claim judgments used as individual-level belief context.
3. **Group belief priors**: WVS-derived belief distributions for demographic groups, organized with the BeliefSim taxonomy.

The code evaluates whether these belief signals improve human-LLM alignment and whether models remain stable under counterfactual demographic swaps.

## Repository Layout

The public code repository is:

```text
https://github.com/MichiganNLP/belief-sim
```

Expected top-level structure:

```text
wvs_misinfo/                   Main BeliefSim experiment code
docs/                          GitHub Pages project website
beliefsim_hf_dataset/          Hugging Face dataset release package, if included
analysis_outputs/              Optional analysis scripts/outputs, if included
```

## Main Code Paths

The main code in `https://github.com/MichiganNLP/belief-sim` lives under `wvs_misinfo/`. The sections below group the scripts by their role in the BeliefSim experiments.

### Prompt Conditioning and Simulation

These scripts construct model prompts using demographics, observed beliefs, and WVS priors, then run model inference.

```text
wvs_misinfo/stress_testing.py                  PANDORA/Prolific prompt-conditioning pipeline
wvs_misinfo/stress_testing_mist.py             MIST prompt-conditioning pipeline
wvs_misinfo/mist_prompt_conditioning.py        MIST prompt generation and evaluation utilities
wvs_misinfo/prolific_dist_prompting.py         Prolific/PANDORA distributional prompting
wvs_misinfo/prolific_education_script.py       Education-axis prompting utilities
```

Model-specific prompt-conditioning runners:

```text
wvs_misinfo/run_prompt_conditioning_gpt4o.py
wvs_misinfo/run_prompt_conditioning_grok4_fast_reasoning.py
wvs_misinfo/run_prompt_conditioning_deepseek_7b.py
wvs_misinfo/run_prompt_conditioning_llama32_3b.py
wvs_misinfo/run_prompt_conditioning_olmo_7b.py
```

### WVS Belief Priors

These scripts use WVS-derived group belief distributions for demographic-conditioned priors.

```text
wvs_misinfo/wvsbelief_sim.py
wvs_misinfo/wvsbelief_sim_aggr.py
wvs_misinfo/wvsbelief_sim_aggr_hf_gender.py
wvs_misinfo/wvsbelief_sim_aggr_hf_urbrur.py
wvs_misinfo/wvsbelief_sim_aggr_hf_age.py
wvs_misinfo/wvsbelief_sim_aggr_hf_no_gender.py
wvs_misinfo/wvsbelief_sim_aggr_hf_no_urbrur.py
wvs_misinfo/wvsbelief_sim_aggr_hf_age_no.py
```

WVS prior tables are stored as CSVs such as:

```text
wvs_misinfo/gender_alldimensions_wvs.csv
wvs_misinfo/urbrur_alldimensions_wvs.csv
wvs_misinfo/age_alldimensions.csv
wvs_misinfo/education_alldimensions.csv
```

### MIST Experiments

These scripts run MIST-specific belief-prior and no-demographic variants.

```text
wvs_misinfo/mist_sim_wvs.py
wvs_misinfo/mist_sim_wvs_nodemo.py
wvs_misinfo/mist_sim_wvs_only.py
wvs_misinfo/mist_sim_wvs_only_nodemo.py
wvs_misinfo/mist_sim_edu_wvs.py
wvs_misinfo/mist_sim_edu_wvs_nodemo.py
wvs_misinfo/mist_sim_edu_wvs_only.py
wvs_misinfo/mist_sim_edu_wvs_only_nodemo.py
wvs_misinfo/mist_sim_wvs_mistral.py
```

MIST inputs used by the local scripts:

```text
wvs_misinfo/MIST - Sample 1 - Raw Dataset.csv
wvs_misinfo/MIST - Phase 4 - Item Database (January 2020).xlsx
```

### Fine-Tuning and Belief Probing

These scripts support the two-stage BeliefSim-FT setup and related probing/evaluation.

```text
wvs_misinfo/phase_a_beliefs_train.py           Phase A belief modeling
wvs_misinfo/phase_a_beliefs_train_llama.py     Llama Phase A variant
wvs_misinfo/phase_a_beliefs_train_qwen.py      Qwen Phase A variant
wvs_misinfo/phase_a_eval.py                    Phase A evaluation
wvs_misinfo/phase_b_susceptibility_train.py    Phase B susceptibility modeling
wvs_misinfo/phase_b_susceptibility_train_qwen.py
wvs_misinfo/phase_b_susceptibility_mist_llama.py
wvs_misinfo/probe_belief.py                    Belief representation probing
wvs_misinfo/lora_eval.py                       LoRA evaluation utilities
wvs_misinfo/llama_finetune_baseline_eval.py    Llama fine-tuning baseline evaluation
```

### Counterfactual and Shortcut-Reliance Evaluation

These scripts evaluate whether predictions flip when demographic fields are counterfactually changed.

```text
wvs_misinfo/shortcut_reliance.py
wvs_misinfo/stress_testing_eval.py
wvs_misinfo/stress_testing_eval_mist.py
wvs_misinfo/eval.py
wvs_misinfo/eval_urbrur.py
wvs_misinfo/eval_veracity.py
wvs_misinfo/eval_ft_baseline.py
```

Batch scripts used for cluster runs:

```text
wvs_misinfo/run_counterfactual_all_scratch.sh
wvs_misinfo/sbatch_counterfactual_all_turbo.sh
wvs_misinfo/run_counterfactual_grok_turbo.sh
wvs_misinfo/sbatch_counterfactual_grok_turbo.sh
```

### Analysis Scripts

```text
analysis_outputs/analyze_modal_revision_results.py
analysis_outputs/analyze_modal_claim_entropy.py
wvs_misinfo/thematic_topic_analysis.py
```

These scripts aggregate model outputs, compute modal/entropy analyses, and produce tables/figures used in the paper.

## Dataset Release

The anonymized release is available on Hugging Face:

```text
https://huggingface.co/datasets/Angana192/beliefsim
```

It has separate viewer configs for:

```text
claims
judgments
evaluation_instances
evaluation_instances_with_claim_text
observed_beliefs
wvs_group_priors
```

Loading example:

```python
from datasets import load_dataset

claims = load_dataset("Angana192/beliefsim", "claims")
judgments = load_dataset("Angana192/beliefsim", "judgments")
instances = load_dataset("Angana192/beliefsim", "evaluation_instances_with_claim_text")
observed_beliefs = load_dataset("Angana192/beliefsim", "observed_beliefs")
wvs_priors = load_dataset("Angana192/beliefsim", "wvs_group_priors")
```

If the dataset is later moved to the MichiganNLP organization, replace `Angana192/beliefsim` with `MichiganNLP/beliefsim`.

To regenerate the Hugging Face release files from local raw data:

```bash
python3 beliefsim_hf_dataset/scripts/prepare_release.py
```

The release script removes direct identifiers, IP/location metadata, raw timestamps, and free-text notes/comments before writing the public CSV files.

## Installation

Create an environment with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scipy scikit-learn matplotlib seaborn tqdm datasets transformers accelerate torch
```

Some scripts require model-specific dependencies or API credentials, depending on whether the run uses local Hugging Face models, hosted APIs, or cluster GPUs.

## Typical Workflows

Generate or rerun prompt-conditioning outputs:

```bash
python wvs_misinfo/stress_testing.py
python wvs_misinfo/stress_testing_mist.py
```

Run model-specific prompt-conditioning scripts:

```bash
python wvs_misinfo/run_prompt_conditioning_llama32_3b.py
python wvs_misinfo/run_prompt_conditioning_olmo_7b.py
python wvs_misinfo/run_prompt_conditioning_deepseek_7b.py
```

Run counterfactual evaluation on the cluster:

```bash
sbatch wvs_misinfo/sbatch_counterfactual_all_turbo.sh
```

Regenerate the dataset release package:

```bash
python3 beliefsim_hf_dataset/scripts/prepare_release.py
```

## Website

The GitHub Pages website is in `docs/`:

```text
docs/index.html
docs/.nojekyll
docs/assets/
```

GitHub Pages should be configured to serve from the `docs/` folder on the `main` branch.

## Content Note

This project studies real misinformation and rumor examples. Some claims may contain offensive, stigmatizing, or otherwise harmful wording. They are included for research transparency and should be handled with care.

## Citation

If you use BeliefSim, please cite our paper:

```bibtex
@article{borah2026belief,
  title={Belief-Sim: Towards Belief-Driven Simulation of Demographic Misinformation Susceptibility},
  author={Borah, Angana and Khan, Zohaib and Mihalcea, Rada and P{\'e}rez-Rosas, Ver{\'o}nica},
  journal={arXiv preprint arXiv:2603.03585},
  year={2026}
}
```

For questions or collaboration inquiries about our misinformation efforts, contact anganab@umich.edu.
