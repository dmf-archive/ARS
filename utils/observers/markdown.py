import json
from pathlib import Path
from typing import Any, Dict, List

from utils.data import CLStore, StepMetric

class MDLogger:
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.task_type = config["experiment"]["tasks"][0] if config["experiment"]["tasks"] else "unknown"

    def on_train_end(self, store: CLStore):
        history = store.get_full_history()
        if not history:
            return

        # Aggregate epoch-level data from step-level history
        epoch_data = self._aggregate_epochs(history)

        # Generate and save the report
        report = self._generate_report(epoch_data, store)
        report_path = self.output_dir / "summary.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)

    def _aggregate_epochs(self, history: List[StepMetric]) -> List[Dict[str, Any]]:
        epochs = {}
        for metric in history:
            if metric.epoch not in epochs:
                epochs[metric.epoch] = {
                    "epoch": metric.epoch + 1,
                    "learning_rate": metric.learning_rate,
                    "pi": [],
                    "effective_gamma": [],
                    "entropy": [],
                    "train_loss": [],
                    "eval_metrics": metric.eval_metrics, # Takes the last one
                }
            epochs[metric.epoch]['pi'].append(metric.pi)
            epochs[metric.epoch]['effective_gamma'].append(metric.effective_gamma)
            epochs[metric.epoch]['entropy'].append(metric.entropy)
            epochs[metric.epoch]['train_loss'].append(metric.loss)

        # Average the collected lists
        agg_epochs = []
        for epoch_num in sorted(epochs.keys()):
            data = epochs[epoch_num]
            data['pi'] = self._avg_or_none(data['pi'])
            data['effective_gamma'] = self._avg_or_none(data['effective_gamma'])
            data['entropy'] = self._avg_or_none(data['entropy'])
            data['train_loss'] = self._avg_or_none(data['train_loss'])
            agg_epochs.append(data)
        
        return agg_epochs

    def _avg_or_none(self, values: List[float | None]) -> float | None:
        valid_values = [v for v in values if v is not None]
        return sum(valid_values) / len(valid_values) if valid_values else None

    def _generate_report(self, epoch_data: List[Dict[str, Any]], store: CLStore) -> str:
        is_cl = len(store.tasks) > 1
        metric_name = "Perplexity" if "wikitext2" in self.task_type else "Accuracy (%)"
        
        # --- Table Generation ---
        headers = ["Epoch", "Train Loss", "Learning Rate", "PI", "Eff. Gamma", "Entropy"]
        task_names = sorted(store.tasks[0].history[-1].eval_metrics.keys())
        for name in task_names:
            headers.append(f"Valid {name}")
        
        table_header = "| " + " | ".join(headers) + " |"
        table_separator = "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"
        
        table_rows = []
        for data in epoch_data:
            row = f"| {data['epoch']} "
            row += f"| {data['train_loss']:.4f} " if data['train_loss'] is not None else "| N/A "
            row += f"| {data['learning_rate']:.6f} "
            row += f"| {data['pi']:.3f} " if data['pi'] is not None else "| N/A "
            row += f"| {data['effective_gamma']:.3f} " if data['effective_gamma'] is not None else "| N/A "
            row += f"| {data['entropy']:.3f} " if data['entropy'] is not None else "| N/A "
            for name in task_names:
                metric_val = data['eval_metrics'].get(name)
                row += f"| {metric_val:.2f} " if metric_val is not None else "| N/A "
            row += "|"
            table_rows.append(row)
        table_content = "\n".join(table_rows)

        # --- Summary Stats ---
        final_metrics = epoch_data[-1]['eval_metrics']
        best_metric_str = self._get_best_metric_summary(epoch_data, task_names)

        # --- Report Assembly ---
        report = f"""# F3EO-Bench Experiment Report

## Configuration Summary
```toml
{json.dumps(self.config, indent=2)}
```

## Training Results
{table_header}
{table_separator}
{table_content}

## Performance Summary
- **Best Validation Metrics**: {best_metric_str}
- **Final Validation Metrics**: {json.dumps(final_metrics)}
"""
        if is_cl:
            report += self._generate_cl_summary(store)

        return report

    def _get_best_metric_summary(self, epoch_data, task_names):
        summary = []
        for name in task_names:
            is_ppl = 'wikitext2' in name
            metrics = [e['eval_metrics'].get(name) for e in epoch_data if e['eval_metrics'].get(name) is not None]
            if not metrics: continue
            best_val = min(metrics) if is_ppl else max(metrics)
            summary.append(f"{name}: {best_val:.2f}")
        return ", ".join(summary)

    def _generate_cl_summary(self, store: CLStore) -> str:
        # Forgetting calculation
        final_accs = {}
        max_accs = {}
        for task in store.tasks:
            accs = [m.eval_metrics.get(task.task_name) for m in task.history if m.eval_metrics.get(task.task_name) is not None]
            if not accs: continue
            final_accs[task.task_name] = accs[-1]
            max_accs[task.task_name] = max(accs)
        
        forgetting = {name: (max_accs[name] - final_accs[name]) / max_accs[name] for name in final_accs if max_accs.get(name, 0) > 0}
        
        return f"""
## Continual Learning Summary
- **Average Forgetting**: {sum(forgetting.values()) / len(forgetting) * 100:.2f}%
- **Forgetting per Task**: {json.dumps({k: f'{v*100:.2f}%' for k, v in forgetting.items()})}
"""