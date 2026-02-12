# examples/examples.py
"""
Run an R2A2 ReAct agent on the GAIA validation split.

Changes from v1:
  - Downloads and passes GAIA attached files to the agent
  - Improved answer normalisation and matching
  - Richer JSONL output (n_steps, tools_used)
  - Per-level accuracy reporting
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

from datasets import load_dataset
from huggingface_hub import login, snapshot_download

from src.agent import R2A2Agent

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────
SAVE_EVERY = 10
MAX_STEPS = 10          # ReAct loop budget per question
MAX_NEW_TOKENS = 1024   # per LLM generation call


# ═══════════════════════════════════════════════════════════════════════
# Answer normalisation and matching (improved)
# ═══════════════════════════════════════════════════════════════════════
def normalize_answer(text: str) -> str:
    """Aggressive normalisation for GAIA exact-match comparison."""
    text = str(text).strip()
    # Remove surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    text = text.lower()
    # Remove trailing punctuation
    text = re.sub(r"[.,;:!?]+$", "", text)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove $ signs and commas in numbers
    text = text.replace("$", "").replace(",", "")
    return text


def check_match(pred: str, gold: Any) -> bool:
    """Check if prediction matches gold answer (quasi exact match)."""
    pred_norm = normalize_answer(pred)

    # Gold can be a list (multiple acceptable answers)
    golds = gold if isinstance(gold, list) else [gold]

    for g in golds:
        g_norm = normalize_answer(str(g))
        # Exact match after normalisation
        if pred_norm == g_norm:
            return True
        # Numeric comparison (handle "17" vs "17.0")
        try:
            if abs(float(pred_norm) - float(g_norm)) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
        # Check if gold is contained in pred (for short answers)
        if len(g_norm) >= 2 and g_norm in pred_norm and len(pred_norm) < len(g_norm) * 3:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════
# GAIA helpers
# ═══════════════════════════════════════════════════════════════════════
def _get_question_and_gold(item: Dict[str, Any]) -> Tuple[str | None, Any]:
    q = item.get("Question") or item.get("question") or item.get("query")
    gold = (
        item.get("Final answer")
        or item.get("final_answer")
        or item.get("golden_answer")
        or item.get("golden_answers")
        or item.get("answer")
    )
    return q, gold


def _get_file_path(item: Dict[str, Any], data_dir: str) -> str:
    """Get the local path to a GAIA attached file, if any."""
    file_name = item.get("file_name") or item.get("file_path") or ""
    if not file_name:
        return ""

    # file_path in GAIA is relative to the dataset repo root
    # e.g. "2023/validation/<uuid>.pdf"
    candidates = [
        os.path.join(data_dir, file_name),
        os.path.join(data_dir, "2023", "validation", file_name),
        os.path.join(data_dir, "2023", "validation", os.path.basename(file_name)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Search recursively
    base = os.path.basename(file_name)
    for root, dirs, files in os.walk(data_dir):
        if base in files:
            return os.path.join(root, base)

    logger.warning("Attached file not found: %s", file_name)
    return ""


def load_completed_ids(path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ═══════════════════════════════════════════════════════════════════════
# Main run
# ═══════════════════════════════════════════════════════════════════════
def run_gaia(agent: R2A2Agent, limit: int | None = None) -> str:
    """Run GAIA validation split, write JSONL, return output path."""

    # ── Authenticate with HuggingFace ─────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        print("Logged in to HuggingFace.")
    else:
        print(
            "WARNING: HF_TOKEN not set. GAIA is a gated dataset — you need to:\n"
            "  1. Go to https://huggingface.co/datasets/gaia-benchmark/GAIA\n"
            "  2. Accept the terms / request access\n"
            "  3. Set HF_TOKEN in your environment\n"
            "Attempting to proceed (will fail if not cached)..."
        )

    # ── Download dataset files ────────────────────────────────────────
    data_dir = None
    try:
        print("Downloading GAIA dataset (with attached files)...")
        data_dir = snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            token=hf_token,
        )
        print(f"Dataset cached at: {data_dir}")
    except Exception as e:
        print(f"WARNING: Could not download full dataset files: {e}")
        print("Continuing without attached files (some questions will be harder).")

    ds = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_all",
        split="validation",
        trust_remote_code=True,
        token=hf_token,
    )

    # Build output filename from model name (e.g. gpt-4o → gaia_gpt-4o_results.jsonl)
    model_tag = getattr(agent.model, "model_id", "unknown")
    model_tag = model_tag.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(REPO_ROOT, "outputs", f"gaia_{model_tag}_results.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    done_ids = load_completed_ids(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} completed records found, skipping them.")

    n_total = 0
    n_new = 0
    correct = 0
    level_stats: Dict[int, Dict[str, int]] = {}  # level -> {total, correct}
    batch_buffer: list[str] = []

    with open(out_path, "a", encoding="utf-8") as f:
        for item in ds:
            q, gold = _get_question_and_gold(item)
            if not q:
                continue

            item_id = str(item.get("task_id") or item.get("id", n_total))
            level = item.get("Level", item.get("level", 0))

            if item_id in done_ids:
                n_total += 1
                continue

            # ── Get attached file path ────────────────────────────────
            file_path = ""
            if data_dir:
                file_path = _get_file_path(item, data_dir)
            if file_path:
                logger.info("Attached file: %s", file_path)

            # ── Run agent ─────────────────────────────────────────────
            t0 = time.time()
            result = agent.run(q, file_path=file_path)
            elapsed = time.time() - t0

            is_match = check_match(result.answer, gold) if gold else False
            if is_match:
                correct += 1

            # Track per-level stats
            lv = int(level) if level else 0
            if lv not in level_stats:
                level_stats[lv] = {"total": 0, "correct": 0}
            level_stats[lv]["total"] += 1
            if is_match:
                level_stats[lv]["correct"] += 1

            row = {
                "id": item_id,
                "question": q,
                "level": lv,
                "gold": gold,
                "pred": result.answer,
                "match": is_match,
                "n_steps": result.n_steps,
                "tools_used": result.tools_used,
                "bias_flag": result.bias_flag,
                "privacy_flag": result.privacy_flag,
                "audit_trail_len": len(result.audit_trail),
                "elapsed_s": round(elapsed, 1),
            }
            batch_buffer.append(json.dumps(row, ensure_ascii=False) + "\n")

            marker = "✓" if is_match else "✗"
            print(
                f"[{n_total:3d}] L{lv} {marker}  "
                f"steps={result.n_steps}  tools={result.tools_used}  "
                f"time={elapsed:.1f}s"
            )
            print(f"       pred={result.answer!r}")
            print(f"       gold={gold!r}")

            n_new += 1
            n_total += 1

            if n_new % SAVE_EVERY == 0:
                f.writelines(batch_buffer)
                f.flush()
                os.fsync(f.fileno())
                acc_so_far = correct / n_new * 100
                print(
                    f"  >>> Checkpoint: {n_new} new ({n_new + len(done_ids)} total)  "
                    f"accuracy={acc_so_far:.1f}%"
                )
                batch_buffer.clear()

            if limit is not None and n_new >= limit:
                break

        if batch_buffer:
            f.writelines(batch_buffer)
            f.flush()
            os.fsync(f.fileno())

    # ── Summary ───────────────────────────────────────────────────────
    total_done = n_new + len(done_ids)
    acc = (correct / n_new * 100) if n_new > 0 else 0.0
    print(f"\n{'=' * 65}")
    print(f"This run:  {correct}/{n_new} correct ({acc:.1f}%)")
    print(f"Total records on disk: {total_done}")
    print(f"\nPer-level breakdown:")
    for lv in sorted(level_stats):
        s = level_stats[lv]
        lv_acc = s['correct'] / s['total'] * 100 if s['total'] else 0
        print(f"  Level {lv}: {s['correct']}/{s['total']} ({lv_acc:.1f}%)")
    print(f"\nOutput: {out_path}")
    print(f"{'=' * 65}")

    return out_path


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run R2A2 agent on GAIA benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen2.5-7B-Instruct",
        help=(
            "Model to use. Examples:\n"
            "  Qwen/Qwen2.5-7B-Instruct   (HuggingFace, needs GPU)\n"
            "  gpt-4o                       (OpenAI, needs OPENAI_API_KEY)\n"
            "  gpt-4o-mini                  (OpenAI, needs OPENAI_API_KEY)\n"
            "  o3-mini                      (OpenAI reasoning)\n"
            "  gemini-2.0-flash             (Google, needs GOOGLE_API_KEY)\n"
            "  gemini-2.5-pro-preview-05-06 (Google, strongest)"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                        help="Max ReAct steps per question (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Max tokens per LLM call (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run first N questions (for testing)")
    args = parser.parse_args()

    # ── Create model using the factory ────────────────────────────────
    from src.api_llm import create_llm
    model = create_llm(
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(f"Using model: {args.model}")

    agent = R2A2Agent(model=model, max_steps=args.max_steps)
    out_path = run_gaia(agent, limit=args.limit)
    print(f"\nSaved results to: {out_path}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
