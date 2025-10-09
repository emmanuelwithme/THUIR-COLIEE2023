from __future__ import annotations

import platform
from pathlib import Path

import torch


def _cpu_name() -> str:
    architecture = (getattr(platform.uname(), "machine", "") or "arch").strip()

    if platform.system() == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            try:
                for line in cpuinfo.read_text().splitlines():
                    if "model name" in line:
                        model = line.split(":", 1)[1].strip()
                        return f"{model} ({architecture})"
            except OSError:
                pass

    processor = platform.processor() or "CPU"
    return f"{processor.strip()} ({architecture.strip()})"


def get_device(prefer_gpu: bool = True, gpu_index: int = 0) -> torch.device:
    """Return a torch.device and print the hardware in use."""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_index}")
        print(f"使用 GPU: {torch.cuda.get_device_name(device)}")
        return device

    device = torch.device("cpu")
    print(f"使用 CPU: {_cpu_name()}")
    return device


if __name__ == "__main__":
    get_device(prefer_gpu=True)
    get_device(prefer_gpu=False)
