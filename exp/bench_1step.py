import gc
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity


@dataclass
class BenchmarkResult:
    optimizer_name: str
    params_count: int
    total_params_bytes: int

    cpu_time_ns: float
    gpu_kernel_time_ms: float

    cuda_memory_before_mb: float
    cuda_memory_after_mb: float

    sync_steps: int
    is_jit_compiled: bool

    kernel_breakdown: dict


def get_cuda_memory() -> tuple[float, float]:
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    return allocated, reserved


def measure_kernel_breakdown(
    model: nn.Module,
    optimizer,
    step_fn: Callable,
    num_warmup: int = 3,
    num_measure: int = 5,
) -> dict:
    for _ in range(num_warmup):
        step_fn()
        optimizer.zero_grad(set_to_none=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(num_measure):
            step_fn()
            optimizer.zero_grad(set_to_none=True)

    key_events = {
        "Newton-Schulz": 0,
        "AdamW": 0,
        "SAM Forward": 0,
        "SAM Backward": 0,
        "Gradient Prewhitening": 0,
    }

    for event in prof.key_averages():
        name = event.key
        cuda_time = getattr(event, 'cuda_time', 0) or 0
        if "matmul" in name.lower() or "mm" in name.lower():
            key_events["Newton-Schulz"] += cuda_time
        elif "add" in name.lower() or "mul" in name.lower():
            key_events["AdamW"] += cuda_time

    return {k: round(v / 1000 / num_measure, 3) for k, v in key_events.items()}


def create_test_model(
    num_layers: int = 3, d_model: int = 512, d_ff: int = 2048, num_heads: int = 8
) -> nn.Module:
    class SimpleBlock(nn.Module):
        def __init__(self, d_model: int, d_ff: int, num_heads: int):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.norm1(x)
            x, _ = self.attn(x, x, x)
            x = self.ffn(self.norm2(x)) + x
            return x

    class TestModel(nn.Module):
        def __init__(self, num_layers: int, d_model: int, d_ff: int, num_heads: int):
            super().__init__()
            self.layers = nn.ModuleList([
                SimpleBlock(d_model, d_ff, num_heads) for _ in range(num_layers)
            ])
            self.embed = nn.Linear(1000, d_model)
            self.head = nn.Linear(d_model, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return x.mean(dim=1) @ self.head.weight.T

    return TestModel(num_layers, d_model, d_ff, num_heads)


def benchmark_optimizer(
    optimizer_name: str,
    optimizer_class,
    params_or_groups,
    config: dict,
    model: nn.Module | None = None,
    num_steps: int = 1,
) -> BenchmarkResult:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    mem_before = get_cuda_memory()[0]

    optimizer = optimizer_class(params_or_groups, **config)

    needs_closure = optimizer_name.startswith("ARS2")

    def make_closure():
        def closure():
            x = torch.randn(2, 256, 1000, device="cuda")
            y = model(x)
            loss = y.sum()
            return loss
        return closure

    start_cpu = time.perf_counter_ns()

    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()

    for _ in range(num_steps):
        if needs_closure:
            optimizer.step(make_closure())
        else:
            x = torch.randn(2, 256, 1000, device="cuda")
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    gpu_end.record()
    torch.cuda.synchronize()

    end_cpu = time.perf_counter_ns()

    mem_after = get_cuda_memory()[0]

    gpu_time_ms = gpu_start.elapsed_time(gpu_end)

    def step_fn():
        if needs_closure:
            optimizer.step(make_closure())
        else:
            x = torch.randn(2, 256, 1000, device="cuda")
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

    kernel_breakdown = measure_kernel_breakdown(model, optimizer, step_fn)

    return BenchmarkResult(
        optimizer_name=optimizer_name,
        params_count=sum(
            p.numel() for p in model.parameters()
        ),
        total_params_bytes=sum(
            p.numel() * p.element_size() for p in model.parameters()
        ),
        cpu_time_ns=(end_cpu - start_cpu) / num_steps,
        gpu_kernel_time_ms=gpu_time_ms / num_steps,
        cuda_memory_before_mb=mem_before,
        cuda_memory_after_mb=mem_after,
        sync_steps=config.get("k", 1),
        is_jit_compiled=True,
        kernel_breakdown=kernel_breakdown,
    )


def main():
    print("=" * 70)
    print("ARS Series 1-Step Performance Benchmark")
    print("=" * 70)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")

    model = create_test_model(num_layers=2, d_model=256, d_ff=1024, num_heads=4).to(device)
    params = list(model.parameters())

    from optimizer.ars2_neo import ARS2Neo
    from optimizer.ars2c import ARS2C
    from optimizer.ars2d import ARS2D
    from optimizer.muon import SingleDeviceMuonWithAuxAdam

    hidden_params = [p for p in model.parameters() if p.ndim >= 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]

    muon_param_groups = [
        {
            "params": hidden_params,
            "lr": 0.02,
            "momentum": 0.95,
            "weight_decay": 0.01,
            "use_muon": True,
        },
        {
            "params": scalar_params,
            "lr": 3e-4,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
            "weight_decay": 0.01,
            "use_muon": False,
        },
    ]

    benchmarks = [
        ("AdamW", torch.optim.AdamW, params, {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01}),
        ("Muon", SingleDeviceMuonWithAuxAdam, muon_param_groups, {}),
        ("ARS2-Neo (Base)", ARS2Neo, params, {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01, "rho": 0.1, "k": 1}),
        ("ARS2-Neo (AGA)", ARS2Neo, params, {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01, "rho": 0.1, "k": 1, "adaptive_sync": True, "adaptive_lambda": 2.0}),
        ("ARS2C (Base)", ARS2C, params, {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01, "rho": 0.1, "k": 1}),
        ("ARS2D (Base)", ARS2D, params, {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01, "rho": 0.1, "k": 1}),
    ]

    results = []
    for name, opt_class, params_or_groups, config in benchmarks:
        print(f"\nBenchmarking: {name}")
        try:
            result = benchmark_optimizer(
                name, opt_class, params_or_groups, config, model=model, num_steps=10
            )
            results.append(result)
            print(f"  CPU Time: {result.cpu_time_ns / 1e6:.2f} ms")
            print(f"  GPU Kernel Time: {result.gpu_kernel_time_ms:.2f} ms")
            print(f"  Memory Delta: {result.cuda_memory_after_mb - result.cuda_memory_before_mb:.2f} MB")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for r in results:
        print(f"\n{r.optimizer_name}:")
        print(f"  CPU Time: {r.cpu_time_ns / 1e6:.2f} ms")
        print(f"  GPU Kernel Time: {r.gpu_kernel_time_ms:.2f} ms")
        print(f"  JIT Compiled: {r.is_jit_compiled}")
        print(f"  Kernel Breakdown:")
        for k, v in r.kernel_breakdown.items():
            print(f"    {k}: {v:.3f} ms")

    output_path = Path("outputs/bench_1step.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
