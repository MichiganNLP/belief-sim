# BeliefSim

**BeliefSim: Towards Belief-Driven Simulation of Demographic Misinformation Susceptibility**

This repository contains the released code and project website for BeliefSim. The code supports prompt-conditioned misinformation-susceptibility simulation, two-phase belief-aware fine-tuning, and counterfactual demographic evaluation.

- Paper: https://arxiv.org/abs/2603.03585
- Project website: https://michigannlp.github.io/belief-sim/
- Dataset: https://huggingface.co/datasets/Angana192/beliefsim

## Repository Structure

```text
prompt_conditioning.py              BeliefSim-PC prompt-conditioning experiments
phase_1_belief_modeling.py          Phase 1: train a belief-distribution head from WVS priors
phase_2_susceptibility_modeling.py  Phase 2: train susceptibility head using Phase 1 belief adapter
couterfactual.py                    Counterfactual demographic evaluation script
docs/                               GitHub Pages project website
docs/assets/                        Website figures and paper visualizations
```

Note: the counterfactual script is currently named `couterfactual.py` in the repository.

## Code Overview

### Prompt Conditioning

`prompt_conditioning.py` runs BeliefSim-PC style prompt conditioning over a Qualtrics-style misinformation dataset export. It constructs prompts with combinations of:

- demographic context,
- observed same-participant belief judgments,
- WVS-derived group belief priors.

It supports multiple Hugging Face causal language models and runs all prompt-condition combinations for demographic axes such as gender, age, and rural/urban living area.

Example:

```bash
python prompt_conditioning.py \
  --input path/to/qualtrics_export.csv \
  --out outputs/beliefsim_pc \
  --hf-model mistralai/Mistral-7B-Instruct-v0.2 \
  --dems gender age urbrur \
  --wvs-run gender=path/to/gender_alldimensions_wvs.csv \
  --wvs-run age=path/to/age_alldimensions.csv \
  --wvs-run urbrur=path/to/urbrur_alldimensions_wvs.csv
```

Outputs are CSV files named from the output prefix, model, demographic axis, and condition tag.

### Phase 1: Belief Modeling

`phase_1_belief_modeling.py` trains a lightweight belief head on top of a frozen base encoder/LM representation. The target is a WVS response distribution over Likert options for a demographic group and survey question.

The script expects a WVS distribution workbook such as `distributions_wvs_all.xlsx` and saves:

```text
belief_head.pt
tokenizer files
```

Default model settings are defined inside the script.

Example:

```bash
python phase_1_belief_modeling.py
```

If needed, edit the top-level config values in the script, such as `MODEL_NAME`, epochs, batch size, and output directory.

### Phase 2: Susceptibility Modeling

`phase_2_susceptibility_modeling.py` trains a susceptibility classifier using the frozen Phase 1 belief adapter. It reads JSON examples containing target headline labels and optional observed-belief examples.

Example:

```bash
python phase_2_susceptibility_modeling.py \
  --base-model Qwen/Qwen2.5-14B-Instruct \
  --phasea-dir phaseA_belief_model_qwen \
  --train-dir path/to/train_examples.json \
  --eval-dir path/to/eval_examples.json \
  --out-dir outputs/phaseB_qwen \
  --epochs 3 \
  --batch-size 8
```

Useful option:

```bash
--no-beliefs-in-prompt
```

This removes observed-belief examples from the Phase 2 prompt, which is useful for ablations.

### Counterfactual Evaluation

`couterfactual.py` evaluates whether predictions flip under demographic counterfactual swaps. It supports three panels used in the paper:

```text
Panel A: Utility
Panel B: Shortcut reliance
Panel C: Complementarity
```

Example:

```bash
python couterfactual.py \
  --input path/to/qualtrics_export.csv \
  --out outputs/counterfactual_gender.csv \
  --panel B \
  --axis gender \
  --hf-model mistralai/Mistral-7B-Instruct-v0.2 \
  --belief gender=path/to/gender_alldimensions_wvs.csv
```

Supported axes:

```text
gender
age
living_area
education
```

## Dataset

The anonymized BeliefSim dataset release is hosted on Hugging Face:

```text
https://huggingface.co/datasets/Angana192/beliefsim
```

It contains separate configs for claims, participant judgments, evaluation instances, observed beliefs, and WVS group priors.

```python
from datasets import load_dataset

claims = load_dataset("Angana192/beliefsim", "claims")
judgments = load_dataset("Angana192/beliefsim", "judgments")
instances = load_dataset("Angana192/beliefsim", "evaluation_instances_with_claim_text")
observed_beliefs = load_dataset("Angana192/beliefsim", "observed_beliefs")
wvs_priors = load_dataset("Angana192/beliefsim", "wvs_group_priors")
```

If the dataset is later moved to the MichiganNLP organization, replace `Angana192/beliefsim` with `MichiganNLP/beliefsim`.

## Installation

Create a Python environment and install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy torch transformers tqdm scikit-learn datasets openpyxl
```

GPU access is recommended for local Hugging Face model inference and fine-tuning.

## Website

The project website is served from `docs/` through GitHub Pages.

```text
docs/index.html
docs/.nojekyll
docs/assets/
```

GitHub Pages should be configured to deploy from the `docs/` folder on the `main` branch.

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
