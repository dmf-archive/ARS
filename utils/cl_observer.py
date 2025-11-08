"""
轻量级持续学习旁路观测器
零侵入：不改动 TrainingMonitor / ReportGenerator / checkpoint
仅维护 (epoch, task_id) 映射表 + 每任务准确率列表
输出：cl_summary.md + cl_curves.json（人工可读）
"""
import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


class CLMetrics:
    """单 epoch 的多任务指标快照"""
    def __init__(self, epoch: int, task_id: int, train_loss: float,
                 accuracies: list[float],  # 每个任务当前测试准确率
                 pi: float | None = None):
        self.epoch = epoch
        self.task_id = task_id
        self.train_loss = train_loss
        self.accuracies = accuracies
        self.pi = pi
        self.timestamp = time.time()


class CLObserver:
    """
    旁路记录持续学习曲线
    用法：
        cl_obs = CLObserver(output_dir)
        for epoch in range(total_epochs):
            for task_id in range(num_tasks):
                train_one_task(...)
                accs = validate_all_tasks()  # [task0_acc, task1_acc, ...]
                cl_obs.log_epoch(epoch, task_id, loss, accs, pi)
        cl_obs.finalize()
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[CLMetrics] = []
        self.console = Console()

    def log_epoch(self, epoch: int, task_id: int, train_loss: float,
                  accuracies: list[float], pi: float | None = None):
        """记录一个 epoch 的多任务准确率快照"""
        self.history.append(CLMetrics(epoch, task_id, train_loss, accuracies, pi))

    def _forgetting_curve(self, task_acc_lists: list[list[float]]) -> list[float]:
        """
        计算每个任务的遗忘率
        task_acc_lists[i] = [acc_after_epoch0, acc_after_epoch1, ...]
        遗忘率 = (best_acc - final_acc) / best_acc
        """
        return [(max(acc) - acc[-1]) / max(acc) if max(acc) > 0 else 0.0
                for acc in task_acc_lists]

    def _rich_text_plot(self, task_acc_lists: list[list[float]]) -> str:
        """
        纯文本迷你折线，宽度 40 字符，高度 8 行
        返回 Rich 可打印的字符串
        """
        if not task_acc_lists:
            return ""
        max_epochs = max(len(acc) for acc in task_acc_lists)
        lines = ["Task  : " + "·" * 40]
        for t, acc in enumerate(task_acc_lists):
            # 线性插值到 40 点
            idx = [int(i * (len(acc) - 1) / 39) for i in range(40)]
            vals = [acc[i] for i in idx]
            # 映射到 0-7 行
            h = 8
            plot = [" " * 40 for _ in range(h)]
            for x, y in enumerate(vals):
                y_row = int((100 - y) / 100 * (h - 1))
                plot[y_row] = plot[y_row][:x] + "█" + plot[y_row][x + 1:]
            lines.append(f"Task {t}: " + plot[0])
            for row in plot[1:]:
                lines.append("       " + row)
        return "\n".join(lines)

    def finalize(self):
        """生成 CL 专用报告"""
        if not self.history:
            return

        # 按 epoch 聚合每任务准确率
        max_epochs = max(m.epoch for m in self.history) + 1
        num_tasks = len(self.history[0].accuracies)
        task_acc_lists: list[list[float]] = [[] for _ in range(num_tasks)]
        epoch_pis: list[float] = []

        for epoch in range(max_epochs):
            # 取该 epoch 最后一次记录的准确率（通常是训练完当前任务后）
            latest = [m for m in self.history if m.epoch == epoch]
            if not latest:
                continue
            best = latest[-1]  # 最晚记录
            for t in range(num_tasks):
                task_acc_lists[t].append(best.accuracies[t])
            epoch_pis.append(best.pi)

        # 计算遗忘率
        forgetting = self._forgetting_curve(task_acc_lists)

        # 画迷你折线
        plot_str = self._rich_text_plot(task_acc_lists)

        # 生成 Markdown
        md_lines = [
            "# CL Quick Report",
            "",
            "## Final Accuracies per Task",
            "| Task | Final Acc (%) | Forgetting (%) |",
            "|------|---------------|----------------|"
        ]
        for t, (acc, fr) in enumerate(zip([acc[-1] for acc in task_acc_lists], forgetting)):
            md_lines.append(f"| Task {t} | {acc:.2f} | {fr * 100:.2f} |")
        md_lines += [
            "",
            "## Average Metrics",
            f"- **Average Final Acc**: {sum(acc[-1] for acc in task_acc_lists) / num_tasks:.2f}%",
            f"- **Average Forgetting**: {sum(forgetting) / num_tasks * 100:.2f}%",
            "",
            "## Mini Forgetting Curve (Text)",
            "```",
            plot_str,
            "```",
            "",
            "## Raw Curves (JSON)",
            "See `cl_curves.json` for detailed numbers."
        ]

        summary = "\n".join(md_lines)
        (self.output_dir / "cl_summary.md").write_text(summary, encoding="utf-8")

        # 导出 JSON 供后续画图
        curves = {
            "task_acc_lists": task_acc_lists,
            "forgetting": forgetting,
            "epoch_pis": epoch_pis,
            "num_tasks": num_tasks,
            "max_epochs": max_epochs
        }
        (self.output_dir / "cl_curves.json").write_text(json.dumps(curves, indent=2))

        self.console.print("\n[bold green]CL observer finalized.[/bold green]")
        self.console.print(f"Report: {self.output_dir / 'cl_summary.md'}")
        self.console.print(f"Curves: {self.output_dir / 'cl_curves.json'}")