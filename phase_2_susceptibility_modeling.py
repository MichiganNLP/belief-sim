import os
import json
import glob
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

NUM_LIKERT = 10
RNG_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def id_to_label(x: int) -> str:
    return "real" if x == 0 else "fake"


def _read_json_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"{path} must be a JSON list[dict]. Got {type(obj)}")
    return obj


def load_json_path(path_or_dir: str) -> List[Dict[str, Any]]:

    if os.path.isfile(path_or_dir):
        if not path_or_dir.endswith(".json"):
            raise ValueError(f"Expected .json file, got: {path_or_dir}")
        return _read_json_file(path_or_dir)

    if os.path.isdir(path_or_dir):
        paths = sorted(glob.glob(os.path.join(path_or_dir, "*.json")))
        if not paths:
            raise FileNotFoundError(f"No .json files found in directory: {path_or_dir}")
        all_rows: List[Dict[str, Any]] = []
        for p in paths:
            all_rows.extend(_read_json_file(p))
        return all_rows

    raise FileNotFoundError(f"Not a file or directory: {path_or_dir}")


def normalize_gold(x: str) -> str:
    x = (x or "").strip().lower()
    if x in {"real", "factual", "true"}:
        return "real"
    if x in {"fake", "deceptive", "false"}:
        return "fake"
    if "real" in x or "factual" in x or "true" in x:
        # guard "not true" style if present
        if "not true" in x or "not real" in x:
            return "fake"
        return "real"
    if "fake" in x or "deceptive" in x or "false" in x or "misinfo" in x:
        return "fake"
    raise ValueError(f"Unrecognized label text: {x}")




def ex_to_label(ex):
    if ex.get("target_label"):
        g = normalize_gold(ex["target_label"])  
        return 0 if g == "real" else 1
    raise KeyError("Example missing target_label")



def build_prompt(ex: Dict[str, Any], no_beliefs_in_prompt: bool = False) -> str:
    demo_axis = (ex.get("demo_axis") or "").strip()
    demo_value = (ex.get("demo_value") or "").strip()

    persona = ""
    if demo_axis and demo_value:
        persona = f"Persona: {demo_axis} = {demo_value}\n"

    exemplars = ""
    if not no_beliefs_in_prompt:
        b1h = (ex.get("belief1_headline") or "").strip()
        b1l = (ex.get("belief1_label") or "").strip()
        b2h = (ex.get("belief2_headline") or "").strip()
        b2l = (ex.get("belief2_label") or "").strip()
        if b1h:
            exemplars += f"Example 1:\nHeadline: {b1h}\nLabel: {b1l}\n\n"
        if b2h:
            exemplars += f"Example 2:\nHeadline: {b2h}\nLabel: {b2l}\n\n"

    target = (ex.get("target_headline") or "").strip()

    instr = (
        "Task: Given the persona (if any) and the examples (if any), "
        "classify the TARGET headline as real or fake.\n"
        "Answer with exactly one label: real or fake.\n\n"
    )

    return instr + persona + exemplars + f"TARGET:\nHeadline: {target}\nAnswer:"


class FrozenBeliefAdapter(nn.Module):

    def __init__(self, base_model_name: str, phasea_dir: str):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        hidden = self.base.config.hidden_size
        self.head = nn.Linear(hidden, NUM_LIKERT)

        head_path = os.path.join(phasea_dir, "belief_head.pt")
        if not os.path.exists(head_path):
            raise FileNotFoundError(f"Missing belief_head.pt at: {head_path}")
        state = torch.load(head_path, map_location="cpu")
        self.head.load_state_dict(state)

        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)

        last_idx = attention_mask.long().sum(dim=1) - 1
        B = out.last_hidden_state.size(0)
        last_hidden = out.last_hidden_state[torch.arange(B, device=last_idx.device), last_idx, :]

        last_hidden = last_hidden.to(self.head.weight.dtype)
        logits = self.head(last_hidden)                       
        probs = torch.softmax(logits, dim=-1)
        return last_hidden.to(torch.float32), probs.to(torch.float32)


class SusceptibilityHeadModel(nn.Module):
    def __init__(self, base_model_name: str, phasea_dir: str):
        super().__init__()
        self.adapter = FrozenBeliefAdapter(base_model_name, phasea_dir)
        hidden = self.adapter.base.config.hidden_size
        self.susc_head = nn.Linear(hidden + NUM_LIKERT, 2)

    def forward(self, input_ids, attention_mask):
        h, belief_probs = self.adapter(input_ids=input_ids, attention_mask=attention_mask)
        feats = torch.cat([h, belief_probs], dim=-1)  
        logits = self.susc_head(feats)                
        return logits

class PhaseBDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer,
        max_length: int,
        no_beliefs_in_prompt: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        examples: List[Dict[str, Any]] = []
        for ex in rows:
            try:
                prompt = build_prompt(ex, no_beliefs_in_prompt=no_beliefs_in_prompt)
                label = ex_to_label(ex)
            except Exception:
                continue
            if not prompt.strip():
                continue
            examples.append({"prompt": prompt, "label": label})

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["prompt"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        attn = [b["attention_mask"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch])

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attn = nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}



def acc_and_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float((y_true == y_pred).mean())

    f1s = []
    for cls in [0, 1]:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    return {"acc": acc, "macro_f1": float(np.mean(f1s))}


@torch.no_grad()
def evaluate_and_save(model, loader, device, out_csv: str):
    model.eval()

    rows = []
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attn)
        preds = torch.argmax(logits, dim=-1)

        for j in range(len(labels)):
            rows.append({
                "example_id": i * loader.batch_size + j,
                "gold_id": int(labels[j].item()),
                "gold_label": id_to_label(int(labels[j].item())),
                "pred_id": int(preds[j].item()),
                "pred_label": id_to_label(int(preds[j].item())),
                "logit_real": float(logits[j, 0].item()),
                "logit_fake": float(logits[j, 1].item()),
                "correct": int(preds[j] == labels[j]),
            })

    import pandas as pd
    import numpy as np

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    if len(df) == 0:
        return {"acc": 0.0, "macro_f1": 0.0}

    y_true = df["gold_id"].to_numpy()
    y_pred = df["pred_id"].to_numpy()

    # --- accuracy ---
    acc = float((y_true == y_pred).mean())

    # --- macro F1 (same logic as your acc_and_macro_f1) ---
    f1s = []
    for cls in [0, 1]:
        tp = ((y_true == cls) & (y_pred == cls)).sum()
        fp = ((y_true != cls) & (y_pred == cls)).sum()
        fn = ((y_true == cls) & (y_pred != cls)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))

    return {"acc": acc, "macro_f1": macro_f1}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base-model", required=True, type=str)
    ap.add_argument("--phasea-dir", required=True, type=str, help="Folder with belief_head.pt (+ tokenizer saved in Phase A)")
    ap.add_argument("--train-dir", required=True, type=str, help="train_examples.json OR folder of *.json")
    ap.add_argument("--eval-dir", required=True, type=str, help="eval_examples.json OR folder of *.json")
    ap.add_argument("--out-dir", required=True, type=str)

    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--no-beliefs-in-prompt", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    train_rows = load_json_path(args.train_dir)
    eval_rows = load_json_path(args.eval_dir)
    print(f"[LOAD] train rows={len(train_rows)} | eval rows={len(eval_rows)}")

    tok_src = args.phasea_dir if os.path.isdir(args.phasea_dir) else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds = PhaseBDataset(train_rows, tokenizer, args.max_length, no_beliefs_in_prompt=args.no_beliefs_in_prompt)
    eval_ds  = PhaseBDataset(eval_rows, tokenizer, args.max_length, no_beliefs_in_prompt=args.no_beliefs_in_prompt)

    pd.DataFrame(train_ds.examples).to_csv(os.path.join(args.out_dir, "train_phaseB_examples.csv"), index=False)
    pd.DataFrame(eval_ds.examples).to_csv(os.path.join(args.out_dir, "eval_phaseB_examples.csv"), index=False)

    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_loader  = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SusceptibilityHeadModel(args.base_model, args.phasea_dir).to(device)

    optimizer = torch.optim.AdamW(model.susc_head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {ep} train"):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attn)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item()) * input_ids.size(0)

        avg_loss = total / max(1, len(train_ds))
        # metrics = evaluate(model, eval_loader, device)
        # print(f"[EPOCH {ep}] train_loss={avg_loss:.4f} | val_acc={metrics['acc']:.4f} | val_macroF1={metrics['macro_f1']:.4f}")
        eval_csv = os.path.join(args.out_dir, f"eval_predictions_epoch{ep}.csv")
        metrics = evaluate_and_save(model, eval_loader, device, eval_csv)

        print(
            f"[EPOCH {ep}] train_loss={avg_loss:.4f} | "
            f"val_acc={metrics['acc']:.4f} | val_macroF1={metrics['macro_f1']:.4f}"
        )

        if metrics["acc"] > best:
            best = metrics["acc"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_phaseB_head.pt"))
            best_csv = os.path.join(args.out_dir, "eval_predictions_best.csv")
            best_metrics = evaluate_and_save(model, eval_loader, device, best_csv)

            print(
                f"[BEST] acc={best_metrics['acc']:.4f} | "
                f"macroF1={best_metrics['macro_f1']:.4f}"
            )

            print(f"[SAVE] best_phaseB_head.pt (acc={best:.4f})")

    # Save final
    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_phaseB_head.pt"))
    print(f"[DONE] Saved best + final to {args.out_dir}")


if __name__ == "__main__":
    main()

