from collections import deque
from threading import Thread
import time

from omegaconf import OmegaConf
import psutil
import torch

import pufferlib.utils


class Profile:
    SPS: ... = 0
    uptime: ... = 0
    remaining: ... = 0
    eval_time: ... = 0
    env_time: ... = 0
    eval_forward_time: ... = 0
    eval_misc_time: ... = 0
    train_time: ... = 0
    train_forward_time: ... = 0
    learn_time: ... = 0
    train_misc_time: ... = 0

    def __init__(self):
        self.start = time.time()
        self.env = pufferlib.utils.Profiler()
        self.eval_forward = pufferlib.utils.Profiler()
        self.eval_misc = pufferlib.utils.Profiler()
        self.train_forward = pufferlib.utils.Profiler()
        self.learn = pufferlib.utils.Profiler()
        self.train_misc = pufferlib.utils.Profiler()
        self.prev_steps = 0
        self.uptime = 0.0

    def __iter__(self):
        yield "SPS", self.SPS
        yield "uptime", self.uptime
        yield "remaining", self.remaining
        yield "eval_time", self.eval_time
        yield "env_time", self.env_time
        yield "eval_forward_time", self.eval_forward_time
        yield "eval_misc_time", self.eval_misc_time
        yield "train_time", self.train_time
        yield "train_forward_time", self.train_forward_time
        yield "learn_time", self.learn_time
        yield "train_misc_time", self.train_misc_time

    @property
    def epoch_time(self):
        return self.train_time + self.eval_time

    def update(self, data, interval_s=1):
        global_step = data.global_step
        if global_step == 0:
            return True

        uptime = time.time() - self.start
        # Always sync profiler totals every call so downstream (e.g. wandb) is not gated on SPS ticks.
        self.eval_time = data._timers["evaluate"].elapsed
        self.eval_forward_time = self.eval_forward.elapsed
        self.env_time = self.env.elapsed
        self.eval_misc_time = self.eval_misc.elapsed
        self.train_time = data._timers["train"].elapsed
        self.train_forward_time = self.train_forward.elapsed
        self.learn_time = self.learn.elapsed
        self.train_misc_time = self.train_misc.elapsed

        if uptime - self.uptime < interval_s:
            return False

        dt = max(uptime - self.uptime, 1e-8)
        self.SPS = (global_step - self.prev_steps) / dt
        self.prev_steps = global_step
        self.uptime = uptime

        self.remaining = (data.config.total_timesteps - global_step) / max(self.SPS, 1e-8)
        return True


def make_losses():
    return OmegaConf.create(
        dict(
            policy_loss=0,
            value_loss=0,
            entropy=0,
            old_approx_kl=0,
            approx_kl=0,
            clipfrac=0,
            explained_variance=0,
        )
    )


def _sample_gpu_util_percent() -> float:
    """NVML이 torch.cuda.utilization()보다 안정적(일부 환경에서 후자가 항상 0)."""
    try:
        import pynvml

        if not hasattr(_sample_gpu_util_percent, "_nvml_ready"):
            pynvml.nvmlInit()
            _sample_gpu_util_percent._nvml_ready = True  # type: ignore[attr-defined]
        n = pynvml.nvmlDeviceGetCount()
        if n == 0:
            return float(torch.cuda.utilization())
        idx = int(torch.cuda.current_device())
        idx = min(max(idx, 0), n - 1)
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        return float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
    except Exception:
        return float(torch.cuda.utilization())


class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque(maxlen=maxlen)
        self.cpu_util = deque(maxlen=maxlen)
        self.gpu_util = deque(maxlen=maxlen)
        self.gpu_mem = deque(maxlen=maxlen)

        self.delay = delay
        self.stopped = False
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(psutil.cpu_percent())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(mem.active / mem.total)
            if torch.cuda.is_available():
                self.gpu_util.append(_sample_gpu_util_percent())
                free, total = torch.cuda.mem_get_info()
                # 남은 비율이 아니라 “사용 중 VRAM 비율(%)”로 표시
                self.gpu_mem.append(100.0 * (1.0 - (free / max(total, 1))))
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        if getattr(_sample_gpu_util_percent, "_nvml_ready", False):
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
            delattr(_sample_gpu_util_percent, "_nvml_ready")
