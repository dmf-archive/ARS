"""
Extract all experimental metrics from summary.md files
and produce a consolidated report.
"""
import json, re, sys
from pathlib import Path
from typing import Any

ROOT = Path("outputs")


def safe_fmt(v: Any, spec: str = "") -> str:
    """Format a value safely; if it's not numeric, return the string representation."""
    if isinstance(v, (int, float)):
        return format(v, spec)
    return str(v)


def find_last_nonempty(rows, col_idx):
    """Find last non-empty value in a column."""
    for row in reversed(rows):
        v = str(row[col_idx]).strip()
        if v and v.lower() != "nan" and v != "N/A":
            try:
                return float(v)
            except ValueError:
                return v
    return "N/A"


def find_best(rows, col_idx, higher_is_better=True):
    """Find best value in a column."""
    best = None
    for row in rows:
        v = str(row[col_idx]).strip()
        if v and v.lower() != "nan" and v != "N/A":
            try:
                fv = float(v)
                if best is None or (higher_is_better and fv > best) or (not higher_is_better and fv < best):
                    best = fv
            except ValueError:
                pass
    return best


def parse_summary_table(path):
    """Parse a standard (non-grokking) summary.md file."""
    content = Path(path).read_text(encoding="utf-8")
    lines = content.split("\n")

    # Find header line
    header_idx = None
    header_line = None
    for i, line in enumerate(lines):
        if line.startswith("| Epoch |"):
            header_idx = i
            header_line = line
            break

    if header_idx is None:
        return None

    headers = [h.strip() for h in header_line.split("|")[1:-1]]

    # Parse data rows
    rows = []
    for line in lines[header_idx + 2:]:
        line = line.strip()
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= len(headers):
            rows.append(cells[:len(headers)])

    if not rows:
        return None

    # Build column map
    col_map = {h: i for i, h in enumerate(headers)}

    # Extract Performance Summary section
    perf_lines = []
    in_perf = False
    for line in lines:
        if line.startswith("## Performance Summary"):
            in_perf = True
            continue
        if in_perf and line.startswith("## "):
            break
        if in_perf:
            perf_lines.append(line)

    perf_text = "\n".join(perf_lines)

    # Determine task type
    first_task = str(rows[0][col_map.get("Task", 0)]).strip().lower() if "Task" in col_map else ""
    is_cifar = "cifar" in first_task or any("accuracy" in h.lower() for h in headers)
    is_wikitext = "wikitext" in first_task or any("perplexity" in h.lower() for h in headers)

    # Extract metrics
    result = {}

    # Last row metrics
    last_row = rows[-1]
    if "Train Loss" in col_map:
        result["final_train_loss"] = float(last_row[col_map["Train Loss"]])
    if "Eval Loss" in col_map:
        result["final_eval_loss"] = float(last_row[col_map["Eval Loss"]])
    if "Epoch Time (s)" in col_map:
        times = [float(r[col_map["Epoch Time (s)"]]) for r in rows if r[col_map["Epoch Time (s)"]] != "N/A" and r[col_map["Epoch Time (s)"]].strip()]
        result["avg_epoch_time"] = sum(times) / len(times) if times else "N/A"

    # Best metrics (from table or Performance Summary)

    # ---- Extract best eval loss from table (works for both CIFAR and Wikitext-2) ----
    if "Eval Loss" in col_map:
        losses = []
        for row in rows:
            v = str(row[col_map["Eval Loss"]]).strip()
            if v and v != "N/A":
                try:
                    losses.append(float(v))
                except ValueError:
                    pass
        if losses:
            result["best_eval_loss"] = min(losses)  # lower loss is better

    # ---- CIFAR-specific metrics ----
    if is_cifar or "Eval Accuracy" in col_map:
        # Parse from Performance Summary
        m = re.search(r"Best Validation Metrics.*?cifar10 Accuracy: ([\d.]+)", perf_text, re.IGNORECASE)
        if m:
            result["best_acc"] = float(m.group(1))
        # Final metrics from Performance Summary
        m = re.search(r"Final Validation Metrics.*?accuracy[:\"]\s*([\d.]+)", perf_text)
        if m:
            result["final_acc"] = float(m.group(1))
        m = re.search(r"Final Validation Metrics.*?loss[:\"]\s*([\d.]+)", perf_text)
        if m:
            result["final_eval_loss_perf"] = float(m.group(1))

        # Also try to find best/final accuracy from table (more reliable)
        if "Eval Accuracy" in col_map:
            accs = []
            for row in rows:
                v = str(row[col_map["Eval Accuracy"]]).strip()
                if v and v != "N/A":
                    try:
                        accs.append(float(v))
                    except ValueError:
                        pass
            if accs:
                result["best_acc_table"] = max(accs)
                result["final_acc_table"] = float(last_row[col_map["Eval Accuracy"]])

    # ---- Wikitext-2 / perplexity metrics ----
    if "Eval Perplexity" in col_map:
        ppls = []
        for row in rows:
            v = str(row[col_map["Eval Perplexity"]]).strip()
            if v and v != "N/A":
                try:
                    ppls.append(float(v))
                except ValueError:
                    pass
        if ppls:
            result["best_ppl"] = min(ppls)
            result["final_ppl"] = float(last_row[col_map["Eval Perplexity"]])
    # Parse from Performance Summary (fallback if table parsing fails)
    m = re.search(r"Best Validation Metrics.*?Perplexity: ([\d.]+)", perf_text)
    if m:
        result["best_ppl_perf"] = float(m.group(1))
    m = re.search(r"Final Validation Metrics.*?perplexity[:\"]\s*([\d.]+)", perf_text)
    if m:
        result["final_ppl_perf"] = float(m.group(1))

    return result


def parse_grok_summary_compact(path):
    """Parse a grokking milestone-format summary.md file."""
    content = Path(path).read_text(encoding="utf-8")
    lines = content.split("\n")

    # Find configuration for optimizer name
    config_text = "\n".join(lines)
    opt_name = "Unknown"
    m = re.search(r'"name":\s*"([^"]+)"', config_text)
    if m:
        opt_name = m.group(1)

    # Find mode (Base/AGA)
    mode = "Base"
    if "adaptive_sync" in config_text and "true" in config_text.split("adaptive_sync")[1][:20].lower():
        mode = "AGA"

    # Find milestones table
    milestones = {}
    in_milestone = False
    for line in lines:
        if line.startswith("| Milestone |"):
            in_milestone = True
            continue
        if in_milestone:
            if line.strip().startswith("|---"):
                continue  # skip separator line
            if line.startswith("|"):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) >= 6:
                    label = cells[0].strip()
                    try:
                        epoch = int(cells[1])
                    except ValueError:
                        continue
                    train_loss = float(cells[2]) if cells[2] != "N/A" else None
                    eval_acc = float(cells[3]) if cells[3] != "N/A" else None
                    milestones[label] = {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "eval_acc": eval_acc,
                    }
            elif not line.startswith("|"):
                break

    # Find Performance Summary
    best_eval_acc = None
    total_epochs = None
    for line in lines:
        m_best = re.search(r"\*+Best Eval Acc\*+:\s*([\d.]+)%", line)
        if m_best:
            best_eval_acc = float(m_best.group(1))
        m_epoch = re.search(r"\*+Total Epochs\*+:\s*(\d+)", line)
        if m_epoch:
            total_epochs = int(m_epoch.group(1))

    # Extract milestone epochs
    result = {
        "optimizer": opt_name,
        "mode": mode,
        "fitting_epoch": None,
        "grokking_epoch": None,
        "convergence_epoch": None,
        "best_eval_acc": best_eval_acc,
        "total_epochs": total_epochs,
    }

    for label, data in milestones.items():
        if "Train Loss < 0.5" in label:
            result["fitting_epoch"] = data["epoch"]
        if "Eval Acc > 90%" in label or "Grokking" in label:
            result["grokking_epoch"] = data["epoch"]
        if "Eval Acc > 99%" in label or "Full Convergence" in label:
            result["convergence_epoch"] = data["epoch"]

    return result


def main():
    all_results = {}

    # ---- CIFAR-10 experiments ----
    cifar_exps = [
        ("ARS2-Neo (Sync, ρ=0.1)", "lrp_cifar10_ars2_neo_sync_60e_rho01"),
        ("ARS2-Neo (Base)", "lrp_cifar10_ars2_neo_base_100e"),
        ("ARS2-Neo (AGA, λ=2.0)", "lrp_cifar10_ars2_neo_aga_20e"),
        ("AdamW", "lrp_cifar10_adamw_100e"),
        ("Muon", "lrp_cifar10_muon_100e"),
    ]
    print("=" * 120)
    print("CIFAR-10 RAW METRICS (ResNet-18, Batch Size 256)")
    print("=" * 120)
    print(f"{'Optimizer':<30s} {'Best Acc':>10s} {'Final Acc':>10s} {'Final Train Loss':>16s} {'Final Eval Loss':>16s} {'Best Eval Loss':>14s} {'Avg Epoch Time':>14s} {'Gen Gap (Acc)':>14s}")
    print("-" * 120)

    for opt_name, exp_dir in cifar_exps:
        path = ROOT / exp_dir / "summary.md"
        if not path.exists():
            print(f"{opt_name:<30s} {'FILE NOT FOUND':>10s}")
            continue
        r = parse_summary_table(str(path))
        if r is None:
            print(f"{opt_name:<30s} {'PARSE ERROR':>10s}")
            continue

        best_acc = r.get("best_acc") if r.get("best_acc") is not None else r.get("best_acc_table", "N/A")
        final_acc = r.get("final_acc") if r.get("final_acc") is not None else r.get("final_acc_table", "N/A")
        final_train_loss = r.get("final_train_loss", "N/A")
        final_eval_loss = r.get("final_eval_loss_perf") if r.get("final_eval_loss_perf") is not None else r.get("final_eval_loss", "N/A")
        best_eval_loss = r.get("best_eval_loss", "N/A")
        avg_time = r.get("avg_epoch_time", "N/A")

        # Generalization gap: Best Acc - Final Acc (negative = degradation)
        if isinstance(best_acc, (int, float)) and isinstance(final_acc, (int, float)):
            gen_gap = best_acc - final_acc
        else:
            gen_gap = "N/A"

        print(f"{opt_name:<30s} {safe_fmt(best_acc, '>10.2f')} {safe_fmt(final_acc, '>10.2f')} {safe_fmt(final_train_loss, '>16.4f')} {safe_fmt(final_eval_loss, '>16.4f')} {safe_fmt(best_eval_loss, '>14.4f')} {safe_fmt(avg_time, '>14.2f')}s {safe_fmt(gen_gap, '>+13.4f')}")
        all_results[f"cifar_{exp_dir}"] = r

    # ---- Wikitext-2 experiments ----
    wt2_exps = [
        ("AdamW", "lrp_wikitext2_adamw_20e"),
        ("Muon", "lrp_wikitext2_muon_20e"),
        ("ARS2-Neo (Base)", "lrp_wikitext2_ars2_neo_base_20e"),
        ("ARS2-Neo (Sync)", "lrp_wikitext2_ars2_neo_sync_10e"),
        ("ARS2-Neo (AGA)", "lrp_wikitext2_ars2_neo_aga_10e"),
    ]
    print()
    print("=" * 120)
    print("WIKITEXT-2 RAW METRICS (Qwen3 RoPE, 3-layer, Context 255)")
    print("=" * 120)
    print(f"{'Optimizer':<25s} {'Best PPL':>10s} {'Final PPL':>10s} {'Best Eval Loss':>15s} {'Final Eval Loss':>15s} {'Final Train Loss':>16s} {'Avg Time':>10s} {'PPL Gap':>10s}")
    print("-" * 120)

    for opt_name, exp_dir in wt2_exps:
        path = ROOT / exp_dir / "summary.md"
        if not path.exists():
            print(f"{opt_name:<25s} {'FILE NOT FOUND':>10s}")
            continue
        r = parse_summary_table(str(path))
        if r is None:
            print(f"{opt_name:<25s} {'PARSE ERROR':>10s}")
            continue

        best_ppl = r.get("best_ppl") if r.get("best_ppl") is not None else r.get("best_ppl_perf", "N/A")
        final_ppl = r.get("final_ppl") if r.get("final_ppl") is not None else r.get("final_ppl_perf", "N/A")
        final_train_loss = r.get("final_train_loss", "N/A")
        final_eval_loss = r.get("final_eval_loss", "N/A")
        best_eval_loss = r.get("best_eval_loss", "N/A")
        avg_time = r.get("avg_epoch_time", "N/A")

        if isinstance(best_ppl, (int, float)) and isinstance(final_ppl, (int, float)):
            ppl_gap = final_ppl - best_ppl
        else:
            ppl_gap = "N/A"

        print(f"{opt_name:<25s} {safe_fmt(best_ppl, '>10.2f')} {safe_fmt(final_ppl, '>10.2f')} {safe_fmt(best_eval_loss, '>15.4f')} {safe_fmt(final_eval_loss, '>15.4f')} {safe_fmt(final_train_loss, '>16.4f')} {safe_fmt(avg_time, '>10.1f')}s {safe_fmt(ppl_gap, '>+10.2f')}")
        all_results[f"wt2_{exp_dir}"] = r

    # ---- Grokking experiments ----
    grok_exps = [
        ("lrp_grok_adamw_600e", "AdamW", "Base"),
        ("lrp_grok_muon_400e", "Muon", "Base"),
        ("lrp_grok_ars2_neo_base_400e", "ARS2-Neo", "Base"),
        ("lrp_grok_ars2_neo_aga_400e", "ARS2-Neo", "AGA"),
        ("lrp_grok_ars2c_aga_400e", "ARS2C", "AGA"),
        ("lrp_grok_ars2c_scaler_aga_400e", "ARS2C (Scaler)", "AGA"),
        ("lrp_grok_ars2d_base_400e", "ARS2D", "Base"),
        ("lrp_grok_ars2d_aga_400e", "ARS2D", "AGA"),
    ]
    print()
    print("=" * 120)
    print("GROKKING RAW METRICS (Modular Addition p=113, train_frac=0.3)")
    print("=" * 120)
    print(f"{'Optimizer':<25s} {'Fitting Ep':>12s} {'Grokking Ep':>12s} {'Converge Ep':>12s} {'Best Eval Acc':>14s} {'Total Ep':>10s}")
    print("-" * 120)

    for exp_dir, opt_name, opt_mode in grok_exps:
        path = ROOT / exp_dir / "summary.md"
        if not path.exists():
            continue

        opt_label = f"{opt_name} ({opt_mode})" if opt_mode != "Base" else opt_name

        # First try milestone format (newer grokking summaries)
        r2 = parse_grok_summary_compact(str(path))

        # If no milestones found, fall back to standard epoch table (old format)
        if r2 is None or (r2.get("fitting_epoch") is None and r2.get("grokking_epoch") is None):
            r_table = parse_summary_table(str(path))
            if r_table is not None and r_table.get("best_acc_table") is not None:
                best_acc = r_table["best_acc_table"]
                print(f"{opt_label:<25s} {'-':>12s} {'-':>12s} {'-':>12s} {best_acc:>13.2f}% {'600':>10s}")
            elif r_table is not None and r_table.get("best_acc") is not None:
                best_acc = r_table["best_acc"]
                print(f"{opt_label:<25s} {'-':>12s} {'-':>12s} {'-':>12s} {best_acc:>13.2f}% {'600':>10s}")
            else:
                # Parse old format via Performance Summary regex directly
                content = Path(path).read_text(encoding="utf-8")
                m = re.search(r"Best Validation Metrics.*?mod_addition Accuracy: ([\d.]+)", content)
                best_acc = float(m.group(1)) if m else None
                if best_acc is not None:
                    print(f"{opt_label:<25s} {'-':>12s} {'-':>12s} {'-':>12s} {best_acc:>13.2f}% {'600':>10s}")
                else:
                    print(f"{opt_label:<25s} {'-':>12s} {'-':>12s} {'-':>12s} {'N/A':>13s} {'600':>10s}")
            continue

        # Use tuple opt_name for display (more accurate than JSON config name)
        opt_label = f"{opt_name} ({opt_mode})" if opt_mode != "Base" else opt_name
        fit = r2["fitting_epoch"] if r2["fitting_epoch"] is not None else ">600"
        grok = r2["grokking_epoch"] if r2["grokking_epoch"] is not None else ">600"
        conv = r2["convergence_epoch"] if r2["convergence_epoch"] is not None else "N/A"
        best = r2["best_eval_acc"]
        total = r2["total_epochs"] if r2["total_epochs"] is not None else "600"

        if isinstance(best, (int, float)):
            print(f"{opt_label:<25s} {str(fit):>12s} {str(grok):>12s} {str(conv):>12s} {best:>13.2f}% {str(total):>10s}")
        else:
            print(f"{opt_label:<25s} {str(fit):>12s} {str(grok):>12s} {str(conv):>12s} {str(best):>13s} {str(total):>10s}")

    print()
    print("=" * 120)
    print("NOTE: 'Gen Gap' = Final - Best, negative means degradation after peak.")
    print("For Grokking experiments with milestone format, only milestone epochs and best eval acc are extracted.")
    print("CIFAR-10 generalization gap = Best Acc - Final Acc (positive means peak degraded).")
    print("Wikitext-2 generalization gap = Final PPL - Best PPL (positive means degradation).")
    print("=" * 120)


if __name__ == "__main__":
    main()
