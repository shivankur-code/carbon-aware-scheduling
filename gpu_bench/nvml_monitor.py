"""NVML sampling and power-cap control. Degrades gracefully when NVML is unavailable."""

from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path

try:
    import pynvml  # provided by the nvidia-ml-py package

    pynvml.nvmlInit()
    HAVE_NVML = True
except Exception:  # noqa: BLE001 - no driver, no GPU, or package missing
    pynvml = None
    HAVE_NVML = False

FIELDS = ["t", "gpu", "power_w", "energy_j", "temp_c", "sm_mhz", "mem_mhz", "util_pct",
          "power_limit_w", "throttle_reasons"]


def _h(i: int):
    return pynvml.nvmlDeviceGetHandleByIndex(i)


def device_info(indices: list[int]) -> list[dict]:
    if not HAVE_NVML:
        return []
    out = []
    for i in indices:
        h = _h(i)
        name = pynvml.nvmlDeviceGetName(h)
        lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        out.append({
            "index": i,
            "name": name.decode() if isinstance(name, bytes) else name,
            "driver": _s(pynvml.nvmlSystemGetDriverVersion()),
            "default_power_limit_w": pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000,
            "min_power_limit_w": lo / 1000,
            "max_power_limit_w": hi / 1000,
            "memory_total_gb": pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e9,
        })
    return out


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def set_power_cap(indices: list[int], frac_of_default: float) -> list[float]:
    """Set each GPU's power limit to frac x default limit (clamped). Returns watts set.

    Needs root. Tries NVML first, then `nvidia-smi -pl` (with sudo if available).
    """
    if not HAVE_NVML:
        raise RuntimeError("NVML unavailable: cannot set power caps")
    watts = []
    for i in indices:
        info = device_info([i])[0]
        w = min(max(frac_of_default * info["default_power_limit_w"],
                    info["min_power_limit_w"]), info["max_power_limit_w"])
        try:
            pynvml.nvmlDeviceSetPowerManagementLimit(_h(i), int(w * 1000))
        except Exception:  # noqa: BLE001
            cmd = ["nvidia-smi", "-i", str(i), "-pl", f"{w:.0f}"]
            if shutil.which("sudo"):
                cmd = ["sudo", "-n", *cmd]
            subprocess.run(cmd, check=True, capture_output=True)
        got = pynvml.nvmlDeviceGetPowerManagementLimit(_h(i)) / 1000
        if abs(got - w) > 5:
            raise RuntimeError(f"GPU {i}: asked for {w:.0f} W, limit is {got:.0f} W")
        watts.append(got)
    return watts


def _throttle(h) -> int:
    for fn in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
        if hasattr(pynvml, fn):
            try:
                return int(getattr(pynvml, fn)(h))
            except Exception:  # noqa: BLE001, S112 - field unsupported on this GPU
                continue
    return -1


def sample(i: int) -> dict:
    h = _h(i)
    try:
        energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0  # mJ -> J
    except Exception:  # noqa: BLE001 - pre-Volta
        energy = float("nan")
    return {
        "t": time.time(),
        "gpu": i,
        "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
        "energy_j": energy,
        "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        "sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
        "mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
        "util_pct": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
        "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0,
        "throttle_reasons": _throttle(h),
    }


class Monitor:
    """Background sampler writing one CSV row per GPU per interval."""

    def __init__(self, indices: list[int], path: Path, interval_s: float = 0.5):
        self.indices, self.path, self.interval = indices, path, interval_s
        self.rows: list[dict] = []
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            if HAVE_NVML:
                for i in self.indices:
                    self.rows.append(sample(i))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thr.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thr.join()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)
