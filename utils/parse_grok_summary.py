import sys
import re
from pathlib import Path


def parse_grok_summary(path: str) -> None:
    content = Path(path).read_text(encoding="utf-8")

    # Find the table: look for the header row and data rows
    lines = content.split("\n")

    # Locate the training results table
    table_start = None
    header_line = None
    sep_line = None
    for i, line in enumerate(lines):
        if line.startswith("| Epoch |"):
            table_start = i
            header_line = line
            sep_line = lines[i + 1] if i + 1 < len(lines) else ""
            break

    if table_start is None:
        print("ERROR: No table found starting with '| Epoch |'")
        return

    # Parse header to find column indices
    headers = [h.strip() for h in header_line.split("|")[1:-1]]

    # Find key column indices
    col_map = {}
    for idx, h in enumerate(headers):
        col_map[h] = idx

    required_cols = ["Epoch", "Eval Accuracy", "Train Loss", "Eval Loss"]
    for col in required_cols:
        if col not in col_map:
            print(f"WARNING: Column '{col}' not found in headers: {headers}")

    # Parse data rows
    rows = []
    for line in lines[table_start + 2:]:
        line = line.strip()
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= len(headers):
            rows.append(cells[:len(headers)])

    if not rows:
        print("ERROR: No data rows found")
        return

    print(f"\n{'='*80}")
    print(f"File: {path}")
    print(f"Total epochs parsed: {len(rows)}")
    print(f"{'='*80}")

    # Extract epoch and eval accuracy
    epoch_idx = col_map.get("Epoch", 0)
    eval_acc_idx = col_map.get("Eval Accuracy", None)
    train_loss_idx = col_map.get("Train Loss", None)
    eval_loss_idx = col_map.get("Eval Loss", None)
    train_acc_idx = col_map.get("Eval Train_accuracy", None)
    pi_idx = col_map.get("PI", None)
    grad_norm_idx = col_map.get("Grad Norm", None)

    # Find milestones
    milestones = {
        "train_loss_lt_0_5": None,
        "train_loss_lt_0_1": None,
        "eval_acc_gt_10": None,
        "eval_acc_gt_50": None,
        "eval_acc_gt_80": None,
        "eval_acc_gt_90": None,
        "eval_acc_gt_95": None,
        "eval_acc_gt_98": None,
        "eval_acc_gt_99": None,
        "train_acc_gt_99": None,
    }

    # Also track best eval acc
    best_eval_acc = 0.0
    best_eval_acc_epoch = 0

    for row in rows:
        epoch = int(row[epoch_idx])

        if eval_acc_idx is not None:
            try:
                eval_acc = float(row[eval_acc_idx])
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_eval_acc_epoch = epoch

                for threshold, key in [(10, "eval_acc_gt_10"), (50, "eval_acc_gt_50"),
                                        (80, "eval_acc_gt_80"), (90, "eval_acc_gt_90"),
                                        (95, "eval_acc_gt_95"), (98, "eval_acc_gt_98"),
                                        (99, "eval_acc_gt_99")]:
                    if eval_acc >= threshold and milestones[key] is None:
                        milestones[key] = (epoch, eval_acc, row)
            except (ValueError, IndexError):
                pass

        if train_loss_idx is not None:
            try:
                tl = float(row[train_loss_idx])
                if tl < 0.5 and milestones["train_loss_lt_0_5"] is None:
                    milestones["train_loss_lt_0_5"] = (epoch, tl, row)
                if tl < 0.1 and milestones["train_loss_lt_0_1"] is None:
                    milestones["train_loss_lt_0_1"] = (epoch, tl, row)
            except (ValueError, IndexError):
                pass

        if train_acc_idx is not None:
            try:
                ta = float(row[train_acc_idx])
                if ta >= 99 and milestones["train_acc_gt_99"] is None:
                    milestones["train_acc_gt_99"] = (epoch, ta, row)
            except (ValueError, IndexError):
                pass

    # Print milestones
    print(f"\n{'─'*80}")
    print(f"  KEY MILESTONES")
    print(f"{'─'*80}")

    milestone_labels = [
        ("train_loss_lt_0_5",  "Train Loss < 0.5   (Fitting)"),
        ("train_loss_lt_0_1",  "Train Loss < 0.1   (Overfit)"),
        ("train_acc_gt_99",    "Train Acc > 99%    (Memorization)"),
        ("eval_acc_gt_10",     "Eval Acc > 10%     (Early Signal)"),
        ("eval_acc_gt_50",     "Eval Acc > 50%     (Above Chance)"),
        ("eval_acc_gt_80",     "Eval Acc > 80%     (Strong Signal)"),
        ("eval_acc_gt_90",     "Eval Acc > 90%     (Grokking)"),
        ("eval_acc_gt_95",     "Eval Acc > 95%     (Near Convergence)"),
        ("eval_acc_gt_98",     "Eval Acc > 98%     (Convergence)"),
        ("eval_acc_gt_99",     "Eval Acc > 99%     (Full Convergence)"),
    ]

    for key, label in milestone_labels:
        val = milestones[key]
        if val is not None:
            epoch, metric, row = val
            extra = ""
            if train_loss_idx is not None:
                extra += f" | Train Loss: {row[train_loss_idx]}"
            if eval_loss_idx is not None:
                extra += f" | Eval Loss: {row[eval_loss_idx]}"
            if pi_idx is not None:
                extra += f" | PI: {row[pi_idx]}"
            if grad_norm_idx is not None:
                extra += f" | Grad Norm: {row[grad_norm_idx]}"
            print(f"  Epoch {epoch:>4d}: {label:35s} ({metric:.2f}){extra}")
        else:
            print(f"  {'─':>4s}  : {label:35s} (NOT REACHED)")

    print(f"\n  Best Eval Acc: {best_eval_acc:.2f}% at Epoch {best_eval_acc_epoch}")
    print(f"{'─'*80}")

    # Print last 5 rows for reference
    print(f"\n  Last 5 epochs:")
    for row in rows[-5:]:
        epoch = int(row[epoch_idx])
        parts = [f"Epoch {epoch:>4d}"]
        if eval_acc_idx is not None:
            parts.append(f"Eval Acc: {row[eval_acc_idx]:>6s}")
        if train_loss_idx is not None:
            parts.append(f"Train Loss: {row[train_loss_idx]:>8s}")
        if eval_loss_idx is not None:
            parts.append(f"Eval Loss: {row[eval_loss_idx]:>8s}")
        print(f"    {' | '.join(parts)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m utils.parse_grok_summary <path_to_summary.md>")
        sys.exit(1)
    parse_grok_summary(sys.argv[1])
