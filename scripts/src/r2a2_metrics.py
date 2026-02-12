# Compute GAIA "Section 5.2-style" proxy metrics programmatically from your JSONL logs.

import json
import numpy as np
from pathlib import Path

FILES = {
    "Qwen2.5-7B-Instruct": "gaia_Qwen_Qwen2.5-7B-Instruct_results.jsonl",
    "GPT-4o-mini": "gaia_gpt-4o-mini_results.jsonl",
    "GPT-4o": "gaia_gpt-4o_results.jsonl",
    "Gemini-2.0-flash": "gaia_gemini-2.0-flash_results.jsonl"
}

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# 1) Load all records + collect global distributions for proxy normalization
all_steps = []
all_audit = []
data = {}

for model, path in FILES.items():
    rows = load_jsonl(path)
    data[model] = rows
    for r in rows:
        all_steps.append(safe_float(r.get("n_steps", 0)))
        all_audit.append(safe_float(r.get("audit_trail_len", 0)))

all_steps = np.array(all_steps, dtype=float)
all_audit = np.array(all_audit, dtype=float)

# Use global 95th percentile constants (reviewer-friendly, stable, not per-model)
S = float(np.percentile(all_steps, 95)) if len(all_steps) else 1.0
A = float(np.percentile(all_audit, 95)) if len(all_audit) else 1.0
S = max(S, 1.0)
A = max(A, 1.0)

def clamp01(x): 
    return max(0.0, min(1.0, x))

def compute_metrics(rows):
    n = len(rows)
    if n == 0:
        return {}

    # episode-level values
    rc = np.array([1.0 if r.get("match", False) else 0.0 for r in rows], dtype=float)
    steps = np.array([safe_float(r.get("n_steps", 0)) for r in rows], dtype=float)
    bias = np.array([1.0 if r.get("bias_flag", False) else 0.0 for r in rows], dtype=float)
    privacy = np.array([1.0 if r.get("privacy_flag", False) else 0.0 for r in rows], dtype=float)
    audit = np.array([safe_float(r.get("audit_trail_len", 0)) for r in rows], dtype=float)

    # proxies (per episode)
    ts = np.array([clamp01(s / S) for s in steps], dtype=float)            # TS_proxy
    al = np.array([clamp01(a / A) for a in audit], dtype=float)            # AL_proxy
    bm = 1.0 - bias                                                        # BM
    pi = 1.0 - privacy                                                     # PI

    # aggregates
    RC = float(rc.mean())
    TS = float(ts.mean())
    BM = float(bm.mean())
    PI = float(pi.mean())
    AL = float(al.mean())

    # indices (equal weights)
    RRI = float((RC + TS + BM + PI + AL) / 5.0)
    CoT_RS_proxy = float((TS + BM + PI + AL) / 4.0)

    # RRC proxy: solved AND no bias/privacy flags AND has audit trail
    rrc_proxy = np.array([
        1.0 if (r.get("match", False) and (not r.get("bias_flag", False)) and (not r.get("privacy_flag", False)) and safe_float(r.get("audit_trail_len", 0)) > 0)
        else 0.0
        for r in rows
    ], dtype=float)
    RRC_proxy = float(rrc_proxy.mean())

    # handy extras
    tool_used = np.array([1.0 if r.get("tools_used", False) else 0.0 for r in rows], dtype=float)
    tools_calls = np.array([safe_float(r.get("tools_calls", 0), 0.0) for r in rows], dtype=float)  # if present
    elapsed = np.array([safe_float(r.get("elapsed_s", 0), 0.0) for r in rows], dtype=float)

    out = {
        "n": n,
        "RC_exact_match": RC,
        "TS_proxy": TS,
        "BM": BM,
        "PI": PI,
        "AL_proxy": AL,
        "RRI": RRI,
        "CoT_RS_proxy": CoT_RS_proxy,
        "RRC_proxy": RRC_proxy,
        "avg_steps": float(steps.mean()),
        "tool_use_rate": float(tool_used.mean()),
        "avg_elapsed_s": float(elapsed.mean()),
    }
    if np.any(tools_calls):
        out["avg_tool_calls"] = float(tools_calls.mean())
    return out

results = {m: compute_metrics(rows) for m, rows in data.items()}

print(f"Global normalization constants: S (P95 steps) = {S:.2f}, A (P95 audit_len) = {A:.2f}\n")

# 2) Print a LaTeX table for the "second table"
latex_rows = []
for model in ["Qwen2.5-7B-Instruct","GPT-4o-mini","GPT-4o","Gemini-2.0-flash"]:
    r = results[model]
    latex_rows.append(
        f"{model} & {r['TS_proxy']:.3f} & {r['BM']:.3f} & {r['PI']:.3f} & {r['AL_proxy']:.3f} & "
        f"{r['RRI']:.3f} & {r['CoT_RS_proxy']:.3f} & {r['RRC_proxy']:.3f} & -- \\\\"
    )


# 3) Also dump a machine-readable JSON summary if you want it
summary_json_path = "gaia_responsible_metrics_summary.json"
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump({"S_P95_steps": S, "A_P95_audit_len": A, "results": results}, f, indent=2)

print("\nSaved JSON summary to:", summary_json_path)