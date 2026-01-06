# phase_a_train.py
import ast
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm



# -------------------------
# Config
# -------------------------
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MAX_LEN = 256
NUM_LIKERT = 10 

DEMOS = [
    ("gender", "female", "Female_distribution"),
    ("gender", "male", "Male_distribution"),
    ("urbrur", "rural", "Rural_distribution"),
    ("urbrur", "urban", "Urban_distribution"),
    ("age", "younger", "Younger_distribution"),
    ("age", "older", "Older_distribution"),
    ("educatio", "high", "higher_distribution"),
    ("education", "low", "lower_distribution"),
]

VAL_FRAC = 0.2
RNG_SEED = 13  # ensures train/val split is reproducible across train/eval scripts


# -------------------------
# Utilities
# -------------------------
def counts_to_probs_and_scale(count_dict: Dict[Any, Any]):
    """
    Convert e.g. {-2: 8, 1: 431, 2: 321, 3: 265, 4: 181}
    -> (probs[10], max_scale)

    - Ignores keys <= 0 (missing / DK / refuse).
    - max_scale = max positive key (1..10).
    - Probs normalized over 1..max_scale, remaining dims = 0.
    """
    vec = np.zeros(NUM_LIKERT, dtype=np.float32)

    positive_keys = []
    for k in count_dict.keys():
        if isinstance(k, str):
            if not k.lstrip("-").isdigit():
                continue
            kk = int(k)
        else:
            kk = int(k)
        if kk > 0:
            positive_keys.append(kk)

    if not positive_keys:
        return np.ones(NUM_LIKERT, dtype=np.float32) / NUM_LIKERT, NUM_LIKERT

    max_scale = max(positive_keys)
    total = 0.0

    for k, v in count_dict.items():
        if isinstance(k, str):
            if not k.lstrip("-").isdigit():
                continue
            kk = int(k)
        else:
            kk = int(k)

        if 1 <= kk <= max_scale:
            vec[kk - 1] += float(v)
            total += float(v)

    if total == 0:
        vec[:max_scale] = 1.0 / max_scale
        return vec, max_scale

    vec /= total
    return vec, max_scale

class WVSDistDataset(Dataset):
    """
    Flatten the Excel into:
        (demo_type, demo_val, question_text, probs[1..10], max_scale)
    and split by question ID so val questions are unseen.
    """

    def __init__(self, xlsx_path: str, tokenizer, split: str = "train",
                 val_frac: float = VAL_FRAC, rng_seed: int = RNG_SEED):
        assert split in {"train", "val"}
        self.tokenizer = tokenizer

        df = pd.read_excel(xlsx_path)

        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            qid = str(row["WVS_Questions"])
            qtext = str(row["Question"])

            for demo_type, demo_val, col in DEMOS:
                dist_str = row.get(col, None)
                if not isinstance(dist_str, str):
                    continue
                try:
                    counts = ast.literal_eval(dist_str)
                    if not isinstance(counts, dict):
                        continue
                except Exception:
                    continue

                probs, max_scale = counts_to_probs_and_scale(counts)
                rows.append(
                    {
                        "demo_type": demo_type,
                        "demo_val": demo_val,
                        "qid": qid,
                        "question": qtext,
                        "probs": probs,       # np.array length 10
                        "max_scale": max_scale,
                    }
                )

        # split by qid so val has unseen questions
        qids = sorted({r["qid"] for r in rows})
        rng = np.random.default_rng(rng_seed)
        rng.shuffle(qids)
        n_val = max(1, int(len(qids) * val_frac))
        val_qids = set(qids[:n_val])

        if split == "train":
            self.rows = [r for r in rows if r["qid"] not in val_qids]
        else:
            self.rows = [r for r in rows if r["qid"] in val_qids]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        max_scale = r["max_scale"]

        prompt = (
            "You are modeling belief distributions for a single demographic group.\n\n"
            f"Demographic ({r['demo_type']}): {r['demo_val']}\n\n"
            "Question:\n"
            f"\"{r['question']}\"\n\n"
            f"There are {max_scale} possible response options, "
            f"numbered 1 through {max_scale}.\n"
            "Predict the probability for each option."
        )

        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        target = torch.tensor(r["probs"], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "target": target,          
            "max_scale": max_scale,    
            "demo_type": r["demo_type"],
        }


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [b["input_ids"] for b in batch]
        attn = [b["attention_mask"] for b in batch]
        targets = torch.stack([b["target"] for b in batch])
        max_scales = torch.tensor([b["max_scale"] for b in batch], dtype=torch.long)

        input_ids = nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attn = nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "target": targets,
            "max_scale": max_scales,
        }


class BeliefHeadModel(nn.Module):

    def __init__(self, base_model_name: str):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        hidden_size = self.base.config.hidden_size
        self.head = nn.Linear(hidden_size, NUM_LIKERT)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # [B, H]

        last_hidden = last_hidden.to(self.head.weight.dtype)

        logits = self.head(last_hidden)                    # [B, 10]
        probs = torch.softmax(logits, dim=-1)
        return probs, logits



def kl_divergence_masked(p_true, p_pred, max_scales, eps=1e-8):
    """
    p_true, p_pred: [B, 10]
    max_scales: [B], per-row max_scale
    KL computed only over 1..max_scale for each row.
    """
    kl_vals = []
    for i in range(p_true.size(0)):
        K = int(max_scales[i].item())
        pt = p_true[i, :K].clamp(min=eps)
        pm = p_pred[i, :K].clamp(min=eps)
        kl = (pt * (pt.log() - pm.log())).sum()
        kl_vals.append(kl)
    return torch.stack(kl_vals).mean()


def js_divergence_masked(p_true, p_pred, max_scales, eps=1e-8):
    js_vals = []
    for i in range(p_true.size(0)):
        K = int(max_scales[i].item())
        pt = p_true[i, :K].clamp(min=eps)
        pm = p_pred[i, :K].clamp(min=eps)
        m = 0.5 * (pt + pm)
        kl1 = (pt * (pt.log() - m.log())).sum()
        kl2 = (pm * (pm.log() - m.log())).sum()
        js_vals.append(0.5 * (kl1 + kl2))
    return torch.stack(js_vals).mean()


def emd_1d_masked(p_true, p_pred, max_scales):
    """
    Earth Mover's Distance on 1D ordinal scale,
    using L1 distance between CDFs, restricted to 1..max_scale.
    """
    emd_vals = []
    for i in range(p_true.size(0)):
        K = int(max_scales[i].item())
        pt = p_true[i, :K]
        pm = p_pred[i, :K]
        cdf_t = torch.cumsum(pt, dim=-1)
        cdf_m = torch.cumsum(pm, dim=-1)
        emd = torch.sum(torch.abs(cdf_t - cdf_m))
        emd_vals.append(emd)
    return torch.stack(emd_vals).mean()


def train_phase_a(
    xlsx_path: str,
    out_dir: str = "phaseA_belief_model_qwen",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 5e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = WVSDistDataset(xlsx_path, tokenizer, split="train")
    val_ds = WVSDistDataset(xlsx_path, tokenizer, split="val")

    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model = BeliefHeadModel(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)

    print(f"#train examples: {len(train_ds)}, #val examples: {len(val_ds)}")
    


    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            max_scale = batch["max_scale"].to(device)
            

            probs, _ = model(input_ids=input_ids, attention_mask=attn)
            loss = kl_divergence_masked(target, probs, max_scale)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

        avg_train_loss = total_loss / len(train_ds)

        model.eval()
        all_true = []
        all_pred = []
        all_max = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                target = batch["target"].to(device)
                max_scale = batch["max_scale"].to(device)
                probs, _ = model(input_ids=input_ids, attention_mask=attn)

                all_true.append(target)
                all_pred.append(probs)
                all_max.append(max_scale)

        all_true = torch.cat(all_true, dim=0)
        all_pred = torch.cat(all_pred, dim=0)
        all_max = torch.cat(all_max, dim=0)

        kl = kl_divergence_masked(all_true, all_pred, all_max).item()
        js = js_divergence_masked(all_true, all_pred, all_max).item()
        emd = emd_1d_masked(all_true, all_pred, all_max).item()

        maj_true_list = []
        maj_pred_list = []
        for i in range(all_true.size(0)):
            K = int(all_max[i].item())
            maj_true_list.append(torch.argmax(all_true[i, :K]).item())
            maj_pred_list.append(torch.argmax(all_pred[i, :K]).item())
        maj_true = torch.tensor(maj_true_list)
        maj_pred = torch.tensor(maj_pred_list)
        maj_acc = (maj_true == maj_pred).float().mean().item()

        print(
            f"Epoch {epoch+1}: "
            f"train_KL={avg_train_loss:.4f}, "
            f"val_KL={kl:.4f}, JS={js:.4f}, EMD={emd:.4f}, "
            f"MajAcc={maj_acc:.3f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.head.state_dict(), os.path.join(out_dir, "belief_head.pt"))
    tokenizer.save_pretrained(out_dir)
    print(f"Saved head + tokenizer to {out_dir}")


if __name__ == "__main__":
    train_phase_a("distributions_wvs_all.xlsx")
