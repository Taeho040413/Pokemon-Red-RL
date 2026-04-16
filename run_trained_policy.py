#!/usr/bin/env python3
"""
학습 없이 저장된 정책만 로드해 포켓몬 레드를 한 환경·한 창(SDL)에서 재생합니다.

로컬 PC 전제: 기본 추론 디바이스는 CPU (--cuda / --device 로만 GPU 사용).
상대 경로(config, runs, ROM, 체크포인트)는 저장소 루트(이 스크립트가 있는 폴더) 우선으로 해석합니다.

  python run_trained_policy.py
  python run_trained_policy.py --checkpoint runs/pokemon-red-003/model_000317.pt
  python run_trained_policy.py --max-steps 5000

환경 변수 POKEMONRED_PLAY_DEVICE=cuda|cpu|mps (선택, --device 보다 우선순위 낮음)
기본 체크포인트 탐색: config.yaml의 train.data_dir / train.exp_id (mtime 최신 model_*.pt).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from pokemonred_puffer.train import (
    DEFAULT_POLICY,
    DEFAULT_REWARD,
    DEFAULT_ROM,
    DEFAULT_WRAPPER,
    find_latest_saved_model,
    load_from_config,
    resolve_resume_checkpoint,
    setup,
)

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"
_ENV_DEVICE = "POKEMONRED_PLAY_DEVICE"
_ENV_TORCH_THREADS = "POKEMONRED_PLAY_TORCH_THREADS"


def _resolve_local_path(path: Path | str, *, root: Path = _PROJECT_ROOT) -> Path:
    """CWD와 무관하게 로컬 클론에서 동작하도록, 상대 경로는 저장소 루트를 우선합니다."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    under_root = (root / p).resolve()
    under_cwd = (Path.cwd() / p).resolve()
    if under_root.exists():
        return under_root
    if under_cwd.exists():
        return under_cwd
    return under_root


def _pick_inference_device(
    *,
    device_arg: str,
    use_cuda_flag: bool,
) -> torch.device:
    """로컬 기본은 CPU. --device / 환경 변수 / --cuda 순으로만 GPU·MPS 사용."""
    raw = device_arg.strip()
    if raw:
        return torch.device(raw)

    env = os.environ.get(_ENV_DEVICE, "").strip().lower()
    if env == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(
            "[run_trained_policy] POKEMONRED_PLAY_DEVICE=cuda 이지만 CUDA 없음 → cpu",
            flush=True,
        )
        return torch.device("cpu")
    if env == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print(
            "[run_trained_policy] POKEMONRED_PLAY_DEVICE=mps 이지만 MPS 없음 → cpu",
            flush=True,
        )
        return torch.device("cpu")
    if env == "cpu":
        return torch.device("cpu")

    if use_cuda_flag:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[run_trained_policy] --cuda 지정이나 CUDA 없음 → cpu", flush=True)

    return torch.device("cpu")


def _apply_local_cpu_thread_defaults(device: torch.device) -> None:
    if device.type != "cpu":
        return
    n = os.environ.get(_ENV_TORCH_THREADS, "").strip()
    if n.isdigit():
        torch.set_num_threads(max(1, int(n)))
        return
    ncpu = os.cpu_count() or 4
    torch.set_num_threads(min(8, max(1, ncpu)))


def _obs_to_batched_torch(obs: Any, device: torch.device) -> Any:
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            t = torch.as_tensor(v, device=device)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            else:
                t = t.unsqueeze(0)
            out[k] = t
        return out
    t = torch.as_tensor(obs, device=device)
    return t.unsqueeze(0) if t.dim() > 0 else t.unsqueeze(0)


def _pick_model_path(args: argparse.Namespace, config: DictConfig) -> Path:
    if args.checkpoint:
        raw = Path(args.checkpoint).expanduser()
        if not raw.is_absolute():
            raw = _resolve_local_path(raw)
        if not raw.exists():
            sys.exit(f"[run_trained_policy] 경로 없음: {raw}")
        model_pt, _ = resolve_resume_checkpoint(raw)
        if model_pt is None:
            sys.exit(f"[run_trained_policy] model_*.pt 를 찾을 수 없음: {raw}")
        return model_pt

    data_dir = _resolve_local_path(str(config.train.get("data_dir", "runs")))
    exp_id = str(getattr(config.train, "exp_id", "") or "").strip()
    found = find_latest_saved_model(data_dir, exp_id, global_fallback=True)
    if found is None:
        sys.exit(
            "[run_trained_policy] 저장된 model_*.pt 가 없습니다. "
            "--checkpoint 로 경로를 지정하세요."
        )
    return found


def _policy_forward(
    policy: torch.nn.Module, obs_torch: Any, lstm_state: Any
) -> tuple[torch.Tensor, Any]:
    """저장본이 RecurrentPolicy(CleanRL) 또는 일반 Policy일 수 있어 분기합니다."""
    with torch.no_grad():
        out = policy(obs_torch, lstm_state)
    if len(out) == 5:
        actions, _lp, _ent, _val, lstm_state = out
    else:
        actions, _lp, _ent, _val = out
        lstm_state = None
    return actions, lstm_state


def main() -> None:
    parser = argparse.ArgumentParser(description="저장 정책으로 레드 단일 창 재생")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="config.yaml 경로",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="model_*.pt 또는 runs/<exp_id> 디렉터리 (미지정 시 최신 자동)",
    )
    parser.add_argument("--rom", type=Path, default=DEFAULT_ROM, help="레드 ROM 경로")
    parser.add_argument(
        "-p",
        "--policy-name",
        default=DEFAULT_POLICY,
        help="config.policies 키 (예: multi_convolutional.MultiConvolutionalPolicy)",
    )
    parser.add_argument(
        "-r",
        "--reward-name",
        default=DEFAULT_REWARD,
        help="config.rewards 키",
    )
    parser.add_argument(
        "-w",
        "--wrappers-name",
        default=DEFAULT_WRAPPER,
        help="config.wrappers 키",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0이면 종료 없음(수동 Ctrl+C). 양수면 그 스텝 후 종료",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="스텝 사이 추가 대기(초). SDL만으로 빠르면 소량 지정",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="추론 디바이스 (예: cpu, cuda). 비우면 기본 cpu (--cuda 와 함께 우선순위 참고)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="가능하면 CUDA로 추론 (로컬 기본은 CPU)",
    )
    args = parser.parse_args()

    config_path = _resolve_local_path(args.config)
    if not config_path.is_file():
        sys.exit(f"[run_trained_policy] config 없음: {config_path}")

    cfg = OmegaConf.load(config_path)
    config = load_from_config(cfg, debug=False)

    device = _pick_inference_device(device_arg=args.device, use_cuda_flag=args.cuda)
    config.train.device = str(device)
    _apply_local_cpu_thread_defaults(device)

    rom_path = _resolve_local_path(args.rom)
    if not rom_path.is_file():
        sys.exit(f"[run_trained_policy] ROM 파일 없음: {rom_path}")

    # 한 개 SDL 창: 헤드리스 끔. 학습용 sqlite/비동기 래퍼는 단일 재생에 불필요.
    config.env.headless = False
    config.env.gb_path = rom_path
    config.train.sqlite_wrapper = False
    config.train.async_wrapper = False

    model_path = _pick_model_path(args, config)
    print(f"[run_trained_policy] 모델: {model_path.resolve()}", flush=True)
    print(f"[run_trained_policy] device: {device}", flush=True)

    policy = torch.load(
        model_path.expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(policy, torch.nn.Module):
        sys.exit("[run_trained_policy] 체크포인트가 nn.Module 이 아닙니다.")
    policy = policy.to(device)
    policy.eval()

    _, env_creator = setup(
        config=config,
        debug=False,
        wrappers_name=args.wrappers_name,
        reward_name=args.reward_name,
        rom_path=rom_path,
        track=False,
        puffer_wrapper=True,
    )
    env = env_creator(
        config.env,
        config.wrappers[args.wrappers_name],
        config.rewards[args.reward_name]["reward"],
    )

    obs, _info = env.reset(seed=config.train.seed)
    lstm_state: Any = None
    total_steps = 0

    print(
        "[run_trained_policy] 재생 시작 (PyBoy SDL 창 1개). 종료: Ctrl+C",
        flush=True,
    )
    try:
        while True:
            obs_torch = _obs_to_batched_torch(obs, device)
            actions, lstm_state = _policy_forward(policy, obs_torch, lstm_state)

            action_np = actions.detach().cpu().numpy()
            if action_np.ndim == 0:
                action_int = int(action_np)
            else:
                action_int = int(action_np.reshape(-1)[0])

            obs, _reward, terminated, truncated, _step_info = env.step(action_int)
            total_steps += 1

            if args.step_delay > 0:
                time.sleep(args.step_delay)

            if terminated or truncated:
                lstm_state = None
                obs, _info = env.reset(seed=config.train.seed)
                print(f"[run_trained_policy] 에피소드 리셋 (step={total_steps})", flush=True)

            if args.max_steps > 0 and total_steps >= args.max_steps:
                print(
                    f"[run_trained_policy] --max-steps={args.max_steps} 도달, 종료",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        print(f"\n[run_trained_policy] 중단 (총 env step ≈ {total_steps})", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
