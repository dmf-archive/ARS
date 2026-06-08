import json
from typing import TYPE_CHECKING

from .base import Callback

if TYPE_CHECKING:
    from utils.data import EpochMetric

    from .context import TrainerContext


class MDLogger(Callback):

    def on_train_begin(self, context: "TrainerContext"):
        pass

    def on_train_end(self, context: "TrainerContext"):
        self._write_report(context)

    def on_epoch_begin(self, context: "TrainerContext"):
        pass

    def on_epoch_end(self, context: "TrainerContext"):
        self._append_epoch_row(context)

    def on_step_begin(self, context: "TrainerContext"):
        pass

    def on_step_end(self, context: "TrainerContext"):
        pass

    def save(self, context: "TrainerContext"):
        pass

    def load(self, context: "TrainerContext") -> bool:
        return False

    def _is_grok(self, context: "TrainerContext") -> bool:
        tasks = context.config.get("experiment", {}).get("tasks", [])
        return "mod_addition" in tasks

    def _append_epoch_row(self, context: "TrainerContext"):
        epoch_history = context.store.get_flat_epoch_history()
        if not epoch_history:
            return
        latest = epoch_history[-1]

        report_path = context.output_dir / "summary.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        is_grok = self._is_grok(context)
        first_write = not report_path.exists() or report_path.stat().st_size == 0

        if is_grok:
            if first_write:
                header = "| Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |\n|-------|------------|----------|-----------|----|-----------|\n"
                with open(report_path, 'w') as f:
                    f.write(header)
            pi_val = latest.avg_pi_obj.raw_pi if latest.avg_pi_obj is not None else "N/A"
            gn_val = f"{latest.grad_norm:.4f}" if latest.grad_norm is not None else "N/A"
            row = (
                f"| {latest.global_epoch + 1} | {latest.avg_train_loss:.4f} | "
                f"{latest.task_metrics.metrics.get('accuracy', 'N/A')} | "
                f"{latest.task_metrics.metrics.get('train_accuracy', 'N/A')} | "
                f"{pi_val} | {gn_val} |\n"
            )
        else:
            if first_write:
                header = "| Epoch | Task | Train Loss | LR | PI | Grad Norm | Epoch Time (s) |\n|-------|------|------------|------|------|-----------|----------------|\n"
                with open(report_path, 'w') as f:
                    f.write(header)
            pi_val = f"{latest.avg_pi_obj.raw_pi:.3f}" if latest.avg_pi_obj is not None else "N/A"
            gn_val = f"{latest.grad_norm:.4f}" if latest.grad_norm is not None else "N/A"
            et_val = f"{latest.epoch_time_s:.2f}" if latest.epoch_time_s is not None else "N/A"
            row = (
                f"| {latest.global_epoch + 1} | {latest.task_name} | {latest.avg_train_loss:.4f} | "
                f"{latest.learning_rate:.6f} | {pi_val} | {gn_val} | {et_val} |\n"
            )

        with open(report_path, 'a') as f:
            f.write(row)

    def _write_report(self, context: "TrainerContext"):
        epoch_history = context.store.get_flat_epoch_history()
        if not epoch_history:
            return

        report = self._generate_report(epoch_history, context.config)
        report_path = context.output_dir / "summary.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)

    def _generate_report(self, epoch_data: list["EpochMetric"], config: dict) -> str:
        task_names = sorted(list(set(e.task_name for e in epoch_data)))

        is_grok = "mod_addition" in task_names

        if is_grok:
            return self._generate_grok_report(epoch_data, config)

        headers = ["Epoch", "Task", "Train Loss", "Min Loss", "Min Step", "LR", "PI", "Eff. Gamma", "Entropy", "Grad Norm", "Epoch Time (s)", "Peak GPU Mem (MB)"]

        diag_keys: set[str] = set()
        for epoch in epoch_data:
            if epoch.diagnostics:
                diag_keys.update(epoch.diagnostics.keys())
        sorted_diag_keys = sorted(list(diag_keys))
        headers.extend([f"Diag {key}" for key in sorted_diag_keys])

        metric_keys: set[str] = set()
        for epoch in epoch_data:
            metric_keys.update(epoch.task_metrics.metrics.keys())
        sorted_metric_keys = sorted(list(metric_keys))
        headers.extend([f"Eval {key.capitalize()}" for key in sorted_metric_keys])

        table_header = "| " + " | ".join(headers) + " |"
        table_separator = "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"

        table_rows = []
        for data in epoch_data:
            row = f"| {data.global_epoch + 1} "
            row += f"| {data.task_name} "
            row += f"| {data.avg_train_loss:.4f} "
            row += f"| {data.min_train_loss:.4f} " if data.min_train_loss is not None else "| N/A "
            row += f"| {data.min_loss_step} " if data.min_loss_step is not None else "| N/A "
            row += f"| {data.learning_rate:.6f} "
            pi_val = getattr(data, 'avg_pi_obj', None)
            if pi_val is not None:
                row += f"| {pi_val.raw_pi:.3f} "
            else:
                row += "| N/A "
            row += f"| {getattr(data, 'avg_effective_gamma', 'N/A')} " if getattr(data, 'avg_effective_gamma', None) is not None else "| N/A "
            row += f"| {data.avg_entropy:.3f} " if data.avg_entropy is not None else "| N/A "
            row += f"| {data.grad_norm:.4f} " if data.grad_norm is not None else "| N/A "
            row += f"| {data.epoch_time_s:.2f} " if data.epoch_time_s is not None else "| N/A "
            row += f"| {data.peak_gpu_mem_mb:.1f} " if data.peak_gpu_mem_mb is not None else "| N/A "

            for key in sorted_diag_keys:
                diag_val = data.diagnostics.get(key) if data.diagnostics else None
                if isinstance(diag_val, float):
                    row += f"| {diag_val:.4f} "
                elif diag_val is not None:
                    row += f"| {diag_val} "
                else:
                    row += "| N/A "

            for key in sorted_metric_keys:
                metric_val = data.task_metrics.metrics.get(key)
                row += f"| {metric_val:.2f} " if isinstance(metric_val, float) else "| N/A "
            row += "|"
            table_rows.append(row)
        table_content = "\n".join(table_rows)

        final_metrics_summary = self._get_final_metrics_summary(epoch_data, task_names)
        best_metric_summary = self._get_best_metric_summary(epoch_data, task_names, sorted_metric_keys)

        report = f"""# ARS-Bench Experiment Report

## Configuration Summary
```json
{json.dumps(config, indent=2)}
```

## Training Results
{table_header}
{table_separator}
{table_content}

## Performance Summary
- **Best Validation Metrics**: {best_metric_summary}
- **Final Validation Metrics**: {final_metrics_summary}
"""
        return report

    def _generate_grok_report(self, epoch_data: list["EpochMetric"], config: dict) -> str:
        grok_epochs = [e for e in epoch_data if e.task_name == "mod_addition"]
        grok_epochs.sort(key=lambda x: x.global_epoch)

        milestone_defs = [
            ("train_loss_lt_0_5", "Train Loss < 0.5   (Fitting)"),
            ("train_loss_lt_0_1", "Train Loss < 0.1   (Overfit)"),
            ("train_acc_gt_99", "Train Acc > 99%    (Memorization)"),
            ("eval_acc_gt_10", "Eval Acc > 10%     (Early Signal)"),
            ("eval_acc_gt_50", "Eval Acc > 50%     (Above Chance)"),
            ("eval_acc_gt_80", "Eval Acc > 80%     (Strong Signal)"),
            ("eval_acc_gt_90", "Eval Acc > 90%     (Grokking)"),
            ("eval_acc_gt_95", "Eval Acc > 95%     (Near Convergence)"),
            ("eval_acc_gt_98", "Eval Acc > 98%     (Convergence)"),
            ("eval_acc_gt_99", "Eval Acc > 99%     (Full Convergence)"),
        ]

        milestone_rows = []
        for key, label in milestone_defs:
            found = None
            for e in grok_epochs:
                if key == "train_loss_lt_0_5" and e.avg_train_loss < 0.5 or key == "train_loss_lt_0_1" and e.avg_train_loss < 0.1 or key == "train_acc_gt_99" and e.task_metrics.metrics.get("train_accuracy", 0) >= 99.0 or key == "eval_acc_gt_10" and e.task_metrics.metrics.get("accuracy", 0) >= 10.0 or key == "eval_acc_gt_50" and e.task_metrics.metrics.get("accuracy", 0) >= 50.0 or key == "eval_acc_gt_80" and e.task_metrics.metrics.get("accuracy", 0) >= 80.0 or key == "eval_acc_gt_90" and e.task_metrics.metrics.get("accuracy", 0) >= 90.0 or key == "eval_acc_gt_95" and e.task_metrics.metrics.get("accuracy", 0) >= 95.0 or key == "eval_acc_gt_98" and e.task_metrics.metrics.get("accuracy", 0) >= 98.0 or key == "eval_acc_gt_99" and e.task_metrics.metrics.get("accuracy", 0) >= 99.0:
                    found = e
                    break

            if found is not None:
                pi_val = found.avg_pi_obj.raw_pi if found.avg_pi_obj is not None else "N/A"
                gn_val = f"{found.grad_norm:.4f}" if found.grad_norm is not None else "N/A"
                milestone_rows.append(
                    f"| {label} | {found.global_epoch + 1} | {found.avg_train_loss:.4f} | "
                    f"{found.task_metrics.metrics.get('accuracy', 'N/A')} | "
                    f"{found.task_metrics.metrics.get('train_accuracy', 'N/A')} | "
                    f"{pi_val} | {gn_val} |"
                )
            else:
                milestone_rows.append(f"| {label} | — | — | — | — | — | — |")

        milestone_table = "\n".join(milestone_rows)

        last_5 = grok_epochs[-5:]
        snapshot_rows = []
        for e in last_5:
            pi_val = e.avg_pi_obj.raw_pi if e.avg_pi_obj is not None else "N/A"
            gn_val = f"{e.grad_norm:.4f}" if e.grad_norm is not None else "N/A"
            snapshot_rows.append(
                f"| {e.global_epoch + 1} | {e.avg_train_loss:.4f} | "
                f"{e.task_metrics.metrics.get('accuracy', 'N/A')} | "
                f"{e.task_metrics.metrics.get('train_accuracy', 'N/A')} | "
                f"{pi_val} | {gn_val} |"
            )
        snapshot_content = "\n".join(snapshot_rows)

        best_eval_acc = 0.0
        best_eval_acc_epoch = 0
        for e in grok_epochs:
            acc = e.task_metrics.metrics.get("accuracy", 0)
            if isinstance(acc, (int, float)) and acc > best_eval_acc:
                best_eval_acc = acc
                best_eval_acc_epoch = e.global_epoch + 1

        report = f"""# ARS-Bench Experiment Report — Grokking Milestone Mode

## Configuration Summary
```json
{json.dumps(config, indent=2)}
```

## Grokking Milestones
| Milestone | Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-----------|-------|------------|----------|-----------|----|-----------|
{milestone_table}

## Performance Summary
- **Best Eval Acc**: {best_eval_acc:.2f}% at Epoch {best_eval_acc_epoch}
- **Total Epochs**: {len(grok_epochs)}

## Last 5 Epochs (Raw Snapshot)
| Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-------|------------|----------|-----------|----|-----------|
{snapshot_content}
"""
        return report

    def _get_best_metric_summary(self, epoch_data: list["EpochMetric"], task_names: list[str], metric_keys: list[str]) -> str:
        summary = []
        for name in task_names:
            for key in metric_keys:
                is_ppl = 'perplexity' in key
                metrics = [e.task_metrics.metrics.get(key) for e in epoch_data if e.task_name == name and e.task_metrics.metrics.get(key) is not None]
                if not metrics: continue
                valid_metrics = [m for m in metrics if isinstance(m, float)]
                if not valid_metrics: continue
                best_val = min(valid_metrics) if is_ppl else max(valid_metrics)
                summary.append(f"{name} {key.capitalize()}: {best_val:.2f}")
        return ", ".join(summary)

    def _get_final_metrics_summary(self, epoch_data: list["EpochMetric"], task_names: list[str]) -> str:
        summary = []
        for name in task_names:
            epochs_for_task = [e for e in epoch_data if e.task_name == name]
            if not epochs_for_task: continue
            last_epoch_for_task = max(epochs_for_task, key=lambda x: x.global_epoch)
            summary.append(f"{name}: {json.dumps(last_epoch_for_task.task_metrics.metrics)}")
        return ", ".join(summary)
