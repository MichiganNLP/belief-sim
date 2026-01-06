import os, re, csv, json, argparse, random
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



def sniff_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except Exception:
            return ","

def load_qualtrics_dual_header(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:

    delim = sniff_delimiter(path)
    raw = pd.read_csv(path, sep=delim, header=None, dtype=str, encoding="utf-8", engine="python")
    if raw.shape[0] < 2:
        raise RuntimeError("File must have at least 2 rows (two headers).")

    row0 = raw.iloc[0].astype(str).tolist()
    row1 = raw.iloc[1].astype(str).tolist()

    seen, labels = {}, []
    for c in row1:
        base, k, idx = c, c, 1
        while k in seen:
            idx += 1
            k = f"{base} [{idx}]"
        seen[k] = True
        labels.append(k)

    qid_map: Dict[str, str] = {}
    q_token = re.compile(r"^\s*(?:QID|Q)\d+\s*$")
    for lbl, head0 in zip(labels, row0):
        qid_map[lbl] = head0.strip() if q_token.match(head0 or "") else ""

    if raw.shape[0] <= 2:
        raise RuntimeError("No data rows (need rows after the two headers).")

    df = raw.iloc[2:].reset_index(drop=True)
    df.columns = labels
    return df, qid_map

def normalize_text(s: Optional[str]) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_label(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())

def find_news_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if "news:" in normalize_text(c).lower()]

def get_first_nonempty(df_row: pd.Series, candidates: List[str]) -> str:
    for c in candidates:
        v = normalize_text(df_row.get(c, ""))
        if v:
            return v
    return ""

def shorten(s: str, n: int = 160) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")




def bucket_education_completed_not(val: str) -> str:
    s = _norm_label(val)
    if not s:
        return ""
    if "not_completed" in s or "not completed" in s:
        return "not_completed"
    if s == "completed":
        return "completed"

    not_completed_keys = [
        "less than high school", "did not complete high school", "no high school",
        "some high school", "incomplete high school", "primary school",
        "elementary school", "middle school", "junior high", "below high school",
    ]
    if any(k in s for k in not_completed_keys):
        return "not_completed"

    completed_keys = [
        "high school graduate", "high school diploma", "ged", "secondary school",
        "some college", "associate", "college", "bachelor", "master", "phd",
        "doctorate", "professional degree", "graduate", "postgraduate", "university",
    ]
    if any(k in s for k in completed_keys):
        return "completed"

    return ""



def build_persona(df_row: pd.Series) -> Dict[str, str]:
    def val(*names: str) -> str:
        return get_first_nonempty(df_row, list(names))

    degree = val(
        "What is the highest degree or level of school you have completed? *If currently enrolled, please indicate the highest degree received.",
        "What is the highest degree or level of school you have completed? \n*If currently enrolled, please indicate the highest degree received.",
        "Highest degree", "Education", "Degree", "Q3",
    )

    persona = {
        "gender": val("What gender do you identify as?", "Gender", "Q2"),
        "age": val("What is your age group?", "Age group", "Q1"),
        "living_area": val("What is your living area?", "Living area", "Q8"),
        "degree": degree,
        "education_bucket": bucket_education_completed_not(degree),
        "employment": val("Employment status", "Q4"),
        "ethnicity": val("Ethnicity", "Q5"),
        "born_country": val("Country of birth", "Q7"),
        "state": val("State", "Which state do you live in?", "US State"),
    }
    return {k: v for k, v in persona.items() if v}



def news_triplets_for_row(
    df: pd.DataFrame, row: pd.Series, news_cols: List[str], qid_map: Dict[str, str]
) -> List[Dict[str, str]]:
    cols = df.columns.tolist()
    trips = []
    for c in news_cols:
        txt = normalize_text(row.get(c, ""))
        if not txt:
            continue
        i = cols.index(c)
        heard_col = cols[i + 1] if i + 1 < len(cols) else ""
        notes_col = cols[i + 2] if i + 2 < len(cols) else ""
        trips.append(
            {
                "news_col": c,
                "news_qid": qid_map.get(c, ""),
                "news_text": txt,
                "heard_val": normalize_text(row.get(heard_col, "")),
                "notes_val": normalize_text(row.get(notes_col, "")),
            }
        )
    return trips


def load_wvs_wide_beliefs(csv_path: str) -> Dict[str, List[str]]:

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8", engine="python")

    qtext_col = next((c for c in df.columns if c.strip().lower() == "question"), None)
    if qtext_col is None:
        qtext_col = next((c for c in df.columns if "question" in c.strip().lower()), None)

    qid_col = next((c for c in df.columns if c.strip().lower() == "question_no"), None)
    if qid_col is None:
        # fallback if stored differently
        qid_col = next((c for c in df.columns if c.strip().lower() in {"qid", "qno", "question id"}), None)

    most_cols = [c for c in df.columns if c.strip().lower().endswith("_most") and c.strip().lower() != "overall_most"]

    table: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        qid = normalize_text(r.get(qid_col, "")) if qid_col else ""
        qtxt = normalize_text(r.get(qtext_col, "")) if qtext_col else ""
        if not qtxt:
            continue

        for mc in most_cols:
            code = normalize_text(r.get(mc, ""))
            if not code or code.lower() == "nan":
                continue
            group = mc.strip().lower().replace("_most", "")  # e.g., female_most -> female
            sent = f'For the survey item {qid} ("{qtxt}"), I would typically choose response option {code}.'
            table.setdefault(group, []).append(sent)

    return table

def load_all_beliefs(assignments: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    --belief key=path (repeatable), where key in {gender,age,living_area,education}
    Loads WVS-wide belief table for each key.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for spec in assignments:
        if "=" not in spec:
            raise RuntimeError(f"--belief must be key=path, got: {spec}")
        key, path = spec.split("=", 1)
        key = key.strip().lower()
        out[key] = load_wvs_wide_beliefs(path.strip())
        print(f"[BELIEFS] Loaded WVS-wide {key} from {path.strip()} groups={len(out[key])}")
    return out

def value_for_dimension(persona: Dict[str, str], dim: str) -> str:
    dim = dim.lower()
    if dim in ("age", "agegroup", "age_group"):
        return persona.get("age", "")
    if dim in ("living_area", "area", "ruralurban"):
        return persona.get("living_area", "")
    if dim == "gender":
        return persona.get("gender", "")
    if dim in ("education", "edu", "education_bucket"):
        return persona.get("education_bucket", "")
    return persona.get(dim, "")

def normalize_axis_value(axis: str, v: str) -> str:
    axis = axis.lower()
    v0 = _norm_label(v)

    if axis == "living_area":
        if "urban" in v0: return "urban"
        if "suburb" in v0: return "suburban"
        if "rural" in v0: return "rural"
        return v0

    if axis == "gender":
        if "female" in v0 or "woman" in v0: return "female"
        if "male" in v0 or "man" in v0: return "male"
        return v0

    if axis in ("education", "edu", "education_bucket"):
        if "not_completed" in v0 or "not completed" in v0: return "not_completed"
        if "completed" in v0: return "completed"
        # if raw degree sneaks in
        b = bucket_education_completed_not(v0)
        return b or v0

    return v0

def map_axis_value_to_wvs_group(axis: str, raw_value: str) -> str:
    v = _norm_label(raw_value)
    axis = axis.lower()

    if axis == "gender":
        if "female" in v or "woman" in v: return "female"
        if "male" in v or "man" in v:     return "male"
        return ""

    if axis in ("living_area", "ruralurban", "area"):
        if "rural" in v: return "rural"
        if "urban" in v: return "urban"
        if "suburban" in v: return "urban"  # fallback
        return ""

    if axis in ("age", "age_group", "agegroup"):
        # map common buckets -> younger/older
        if any(x in v for x in ["18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "18-24", "25-34", "30-44", "30", "31", "32", "33", "34", "35", "40"]):
            return "younger"
        if any(x in v for x in ["45", "46", "47", "48", "49", "50", "55", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "older", "45+", "55-64", "65-74", "75-84"]):
            return "older"
        return ""

    if axis in ("education", "edu", "education_bucket"):
        b = bucket_education_completed_not(v)
        if b == "completed":
            # accept either naming convention in WVS files
            return "completed"
        if b == "not_completed":
            return "not_completed"
        return ""

    return ""

def collect_csv_priors(persona: Dict[str, str], belief_index: Dict[str, Dict[str, List[str]]]) -> List[str]:
    priors: List[str] = []
    for dim, table in belief_index.items():
        val = value_for_dimension(persona, dim)
        if not val:
            continue
        gkey = map_axis_value_to_wvs_group(dim, val)
        if not gkey:
            continue

        if gkey in table:
            priors.extend(table[gkey])
            continue

        if dim == "education":
            if gkey == "completed" and "highedu" in table:
                priors.extend(table["highedu"])
            elif gkey == "not_completed" and "lowedu" in table:
                priors.extend(table["lowedu"])

    return priors



def system_prompt_demo_only(axis: str, persona: Dict[str, str]) -> str:
    axis = axis.lower()
    v = persona.get(axis, "")
    if axis == "education":
        v = persona.get("education_bucket", "") or v
    return (
        "You are predicting whether a person would judge a claim as True Information or Misinformation.\n"
        f"Person attribute (only info you know): {axis} = {v or 'unknown'}.\n"
        "Do not mention being an AI. Follow the output format strictly."
    )

def system_prompt_beliefs_only(belief_lines: List[str]) -> str:
    bel = "\n".join([f"- {x}" for x in belief_lines]) if belief_lines else "(none)"
    return (
        "You are predicting whether a person would judge a claim as True Information or Misinformation.\n"
        "You know the following belief evidence about the person (treat as fixed context; do not debate it):\n"
        f"{bel}\n"
        "Do not mention being an AI. Follow the output format strictly."
    )

def system_prompt_beliefs_plus_demo(axis: str, persona: Dict[str, str], belief_lines: List[str]) -> str:
    axis = axis.lower()
    v = persona.get(axis, "")
    if axis == "education":
        v = persona.get("education_bucket", "") or v
    bel = "\n".join([f"- {x}" for x in belief_lines]) if belief_lines else "(none)"
    return (
        "You are predicting whether a person would judge a claim as True Information or Misinformation.\n"
        f"Person attribute: {axis} = {v or 'unknown'}.\n"
        "Belief evidence about the person (treat as fixed context; do not debate it):\n"
        f"{bel}\n"
        "Do not mention being an AI. Follow the output format strictly."
    )

def news_user_prompt(qid: str, label: str, text: str) -> str:
    return (
        f"[{qid or 'QID?'}] Question: {label}\n"
        f"News text:\n{text}\n\n"
        "From this person's perspective, is the information trustworthy?\n"
        "Choose exactly one:\n"
        "(1) True Information\n"
        "(2) Misinformation\n"
        'Respond in this exact format: "Choice: <number>. <text>" (no extra commentary).'
    )



def swap_value(axis: str, current: str, pool: List[str], rng: random.Random) -> Optional[str]:
    axis = axis.lower()
    cur = normalize_axis_value(axis, current)
    if not cur:
        return None

    if axis == "gender":
        if cur == "male": return "female"
        if cur == "female": return "male"

    if axis == "living_area":
        if cur == "urban": return "rural"
        if cur == "rural": return "urban"
        if cur == "suburban": return "urban"

    if axis in ("education", "edu", "education_bucket"):
        if cur == "completed": return "not_completed"
        if cur == "not_completed": return "completed"

    candidates = [p for p in pool if normalize_axis_value(axis, p) != cur and p]
    return rng.choice(candidates) if candidates else None



def load_hf_chat_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    return model, tok

def call_hf_chat(model, tok, system_msg: str, user_msg: str,
                 max_new_tokens=80, temperature=0.3, top_p=0.95) -> str:
    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


# Swapping

def bucket_human(val: str) -> str:
    t = normalize_text(val).lower()
    if "misinformation" in t or "false" in t: return "MISINFO"
    if "true" in t: return "TRUE"
    if "not sure" in t or "unsure" in t: return "NOT_SURE"
    return "OTHER"

def locate_human_response_col_for_qid(
    df: pd.DataFrame,
    qid_map: Dict[str,str],
    news_cols: List[str],
    target_news_col: str
) -> Optional[str]:

    cols = df.columns.tolist()
    start = cols.index(target_news_col)
    idx_news = news_cols.index(target_news_col)
    end = cols.index(news_cols[idx_news + 1]) if (idx_news + 1) < len(news_cols) else len(cols)
    block = cols[start:end]
    for c in block:
        lc = normalize_text(c).lower()
        if "your responses" in lc or "your response" in lc or "your answer" in lc:
            return c
    return None

def build_panelB_balanced_indices(
    df: pd.DataFrame,
    qid_map: Dict[str,str],
    news_cols: List[str],
    axis: str,
    max_news: int,
    seed: int,
    max_total: int = 800,
) -> List[int]:

    rng = random.Random(seed)
    axis = axis.lower()

    rows_meta = []
    for i in range(len(df)):
        row = df.iloc[i]
        persona = build_persona(row)

        # for education, require bucket
        if axis == "education":
            if "education_bucket" not in persona or not persona.get("education_bucket"):
                continue
            axis_val = persona.get("education_bucket", "")
        else:
            if axis not in persona:
                continue
            axis_val = persona.get(axis, "")

        trips = news_triplets_for_row(df, row, news_cols, qid_map)
        if not trips:
            continue
        trips = trips[:max_news]
        target_trip = trips[-1]
        target_col = target_trip["news_col"]

        ycol = locate_human_response_col_for_qid(df, qid_map, news_cols, target_col)
        if not ycol:
            continue

        hb = bucket_human(row.get(ycol, ""))
        if hb not in {"TRUE","MISINFO","NOT_SURE"}:
            continue

        group = normalize_axis_value(axis, axis_val)
        if not group:
            continue

        rows_meta.append((i, target_trip["news_qid"], group, hb))

    if not rows_meta:
        return []

    meta_df = pd.DataFrame(rows_meta, columns=["row_index","qid","group","hb"])

    chosen = []
    for qid, sub in meta_df.groupby("qid"):
        groups = sub["group"].unique().tolist()
        if len(groups) < 2:
            continue

        pivot = sub.groupby(["group","hb"]).size().unstack(fill_value=0)
        labels = [c for c in pivot.columns if (pivot[c] > 0).sum() >= 2]
        if not labels:
            continue

        min_per_cell = min(int(pivot.loc[g, lab]) for g in pivot.index for lab in labels)
        if min_per_cell <= 0:
            continue

        take = min(min_per_cell, 5)
        for g in pivot.index:
            for lab in labels:
                cell = sub[(sub["group"] == g) & (sub["hb"] == lab)]["row_index"].tolist()
                rng.shuffle(cell)
                chosen.extend(cell[:take])

    rng.shuffle(chosen)
    return chosen[:max_total]




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", choices=["A","B","C"], required=True)
    ap.add_argument("--axis", choices=["gender","age","living_area","education"], required=True)
    ap.add_argument("--max-news", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--belief", action="append", default=[], help="Attach WVS-wide belief CSV as key=path")
    ap.add_argument("--belief-drop-rate", type=float, default=0.0, help="For Panel C: randomly drop this fraction of priors")
    ap.add_argument("--panelC-n", type=int, default=300, help="Max rows for Panel C")
    ap.add_argument("--max-total", type=int, default=800, help="Max rows for Panel B")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    df, qid_map = load_qualtrics_dual_header(args.input)
    news_cols = find_news_columns(df)
    if not news_cols:
        raise RuntimeError("No News: columns found")

    belief_index = load_all_beliefs(args.belief) if args.belief else {}

    model, tok = load_hf_chat_model(args.hf_model)

    # pool for counterfactual swaps (use bucket for education)
    axis_pool = []
    for _, r in df.iterrows():
        p = build_persona(r)
        if args.axis == "education":
            axis_pool.append(p.get("education_bucket", ""))
        else:
            axis_pool.append(p.get(args.axis, ""))
    axis_pool = [normalize_text(v) for v in axis_pool if normalize_text(v)]

    # choose indices depending on panel
    if args.panel == "A":
        indices = list(range(len(df)))
    elif args.panel == "B":
        indices = build_panelB_balanced_indices(
            df, qid_map, news_cols, args.axis, args.max_news, args.seed, max_total=args.max_total
        )
        print(f"[PANEL B] balanced subset size = {len(indices)}")
    else:  # C
        indices = list(range(len(df)))
        rng.shuffle(indices)
        indices = indices[:args.panelC_n]
        print(f"[PANEL C] sample size = {len(indices)}")

    possible_ids = [
        "ResponseId","Response ID","Response Id","_recordId","External Data Reference",
        "ExternalReference","PROLIFIC_ID","prolific_id"
    ]

    out_rows = []
    for i in indices:
        row = df.iloc[i]
        respondent_id = get_first_nonempty(row, possible_ids) or f"row_{i}"
        persona = build_persona(row)

        # ensure axis value exists
        if args.axis == "education":
            if not persona.get("education_bucket"):
                continue
        else:
            if args.axis not in persona:
                continue

        trips = news_triplets_for_row(df, row, news_cols, qid_map)
        if not trips:
            continue
        trips = trips[:args.max_news]
        target_trip = trips[-1]

        # Observed "belief examples" (optional): use previous news items as context.
        observed_beliefs = []
        if len(trips) >= 2:
            for b in trips[:-1]:
                observed_beliefs.append(
                    f"Previously saw: {shorten(b['news_text'], 140)}; Heard: {b['heard_val'] or '—'}."
                )

        # CSV priors (imputed beliefs)
        csv_priors = collect_csv_priors(persona, belief_index)

        belief_lines = observed_beliefs + csv_priors

        # Panel C: degrade beliefs by dropping many priors
        if args.panel == "C" and args.belief_drop_rate > 0 and belief_lines:
            kept = []
            for bl in belief_lines:
                if rng.random() >= args.belief_drop_rate:
                    kept.append(bl)
            belief_lines = kept

        # conditions per panel
        # A,B: demo_only
        # C: beliefs_only_degraded AND beliefs_plus_demo_degraded
        conditions = ["demo_only"] if args.panel in {"A","B"} else ["beliefs_only_degraded","beliefs_plus_demo_degraded"]

        # counterfactual swap value for this axis
        axis_current = persona.get("education_bucket", "") if args.axis == "education" else persona.get(args.axis, "")
        swapped = swap_value(args.axis, axis_current, axis_pool, rng)

        for cond in conditions:
            if cond == "demo_only":
                sys_msg = system_prompt_demo_only(args.axis, persona)
                persona_swapped = dict(persona)
                if swapped:
                    if args.axis == "education":
                        persona_swapped["education_bucket"] = swapped
                    else:
                        persona_swapped[args.axis] = swapped
                sys_msg_sw = system_prompt_demo_only(args.axis, persona_swapped) if swapped else ""

            elif cond == "beliefs_only_degraded":
                sys_msg = system_prompt_beliefs_only(belief_lines)
                sys_msg_sw = sys_msg  # no demographics present, so swap irrelevant

            else:  # beliefs_plus_demo_degraded
                sys_msg = system_prompt_beliefs_plus_demo(args.axis, persona, belief_lines)
                persona_swapped = dict(persona)
                if swapped:
                    if args.axis == "education":
                        persona_swapped["education_bucket"] = swapped
                    else:
                        persona_swapped[args.axis] = swapped
                sys_msg_sw = system_prompt_beliefs_plus_demo(args.axis, persona_swapped, belief_lines) if swapped else ""

            user_msg = news_user_prompt(
                target_trip["news_qid"],
                normalize_text(target_trip["news_col"]),
                target_trip["news_text"],
            )

            try:
                resp = call_hf_chat(model, tok, sys_msg, user_msg)
            except Exception as e:
                resp = f"[ERROR] {type(e).__name__}: {e}"

            resp_sw = ""
            if swapped and sys_msg_sw:
                try:
                    resp_sw = call_hf_chat(model, tok, sys_msg_sw, user_msg)
                except Exception as e:
                    resp_sw = f"[ERROR] {type(e).__name__}: {e}"

            axis_value = persona.get("education_bucket", "") if args.axis == "education" else persona.get(args.axis, "")
            out_rows.append({
                "panel": args.panel,
                "axis": args.axis,
                "condition": cond,
                "model_id": args.hf_model,
                "row_index": i,
                "respondent_id": respondent_id,
                "target_news_qid": target_trip["news_qid"],
                "target_news_label": normalize_text(target_trip["news_col"]),
                "target_news_text": target_trip["news_text"],
                "axis_value": axis_value,
                "axis_value_swapped": swapped or "",
                "target_choice_raw": resp,
                "target_choice_raw_swapped": resp_sw,
                "belief_lines_count": len(belief_lines),
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"[DONE] wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
