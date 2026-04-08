import functools
import importlib
import os
import sqlite3
from tempfile import NamedTemporaryFile
import time
import uuid
from contextlib import contextmanager, nullcontext

from enum import Enum
from multiprocessing import Queue
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Callable

import gymnasium
import pufferlib
import pufferlib.emulation
import pufferlib.vector
import typer
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

import wandb
from pokemonred_puffer import cleanrl_puffer
from pokemonred_puffer.cleanrl_puffer import CleanPuffeRL
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.wrappers.async_io import AsyncWrapper
from pokemonred_puffer.wrappers.sqlite import SqliteStateResetWrapper

app = typer.Typer(pretty_exceptions_enable=False)

DEFAULT_CONFIG = "config.yaml"
DEFAULT_POLICY = "multi_convolutional.MultiConvolutionalPolicy"
DEFAULT_REWARD = "baseline.ExplorationInteractionRewardEnv"
DEFAULT_WRAPPER = "exploration_interaction"
DEFAULT_ROM = "red.gb"


def find_latest_saved_model(
    data_dir: Path,
    exp_id: str | None,
    *,
    global_fallback: bool = False,
) -> Path | None:
    root = data_dir.expanduser().resolve()
    if not root.is_dir():
        return None
    exp_id = (exp_id or "").strip()

    if exp_id:
        run_dir = root / exp_id
        if run_dir.is_dir():
            picked = _pick_latest_checkpoint_by_mtime(_iter_numeric_model_pts(run_dir))
            if picked is not None:
                return picked
    if global_fallback:
        all_pts = [
            p
            for p in root.rglob("model_*.pt")
            if _model_checkpoint_sort_key(p) >= 0
        ]
        if all_pts:
            return _pick_latest_checkpoint_by_mtime(all_pts)
    return None


def _find_latest_from_config(cfg: DictConfig) -> Path | None:
    data_dir = Path(cfg.train.get("data_dir", "runs"))
    exp_id = str(getattr(cfg.train, "exp_id", "") or "").strip()
    global_fb = bool(cfg.train.get("resume_latest_global", False))
    return find_latest_saved_model(data_dir, exp_id, global_fallback=global_fb)


def _find_latest_default_auto(cfg: DictConfig) -> Path | None:
    data_dir = Path(cfg.train.get("data_dir", "runs"))
    exp_id = str(getattr(cfg.train, "exp_id", "") or "").strip()
    found = find_latest_saved_model(data_dir, exp_id, global_fallback=False)
    if found is None:
        found = find_latest_saved_model(data_dir, exp_id, global_fallback=True)
    return found


def _effective_resume_path(
    checkpoint_cli: Path | None,
    cfg: DictConfig,
    *,
    resume_latest_cli: bool = False,
    fresh_cli: bool = False,
) -> tuple[Path | None, bool, bool]:
    if fresh_cli or bool(cfg.train.get("resume_fresh", False)):
        return None, False, True
    if checkpoint_cli is not None:
        return checkpoint_cli.expanduser(), False, False
    rc = cfg.train.get("resume_checkpoint")
    rc_str = str(rc).strip() if rc is not None else ""
    rc_lower = rc_str.lower()
    if rc_lower in ("latest", "auto"):
        found = _find_latest_from_config(cfg)
        return found, True, False
    if rc is not None and rc_str not in ("", "~", "null", "None"):
        return Path(rc_str).expanduser(), False, False
    if resume_latest_cli or bool(cfg.train.get("resume_latest", False)):
        found = _find_latest_from_config(cfg)
        return found, True, False
    found = _find_latest_default_auto(cfg)
    return found, True, False


def _log_resume_intent(
    resume_src: Path | None,
    resume_auto_attempted: bool,
    model_pt: Path | None,
    *,
    resume_fresh: bool,
) -> None:
    """환경/정책 생성 전에 호출 — 긴 vecenv 스폰·Rich 이후에 밀려 안 보이는 것 방지."""
    if resume_fresh:
        print("[resume] 처음부터 학습합니다 (--fresh 또는 train.resume_fresh).", flush=True)
        return
    if resume_auto_attempted and resume_src is None:
        print("[resume] runs 아래 model_*.pt 를 찾지 못했습니다. 처음부터 학습합니다.", flush=True)
        return
    if resume_src is not None:
        resolved = resume_src.expanduser().resolve()
        if resume_auto_attempted:
            print(f"[resume] 최신 체크포인트 자동 선택: {resolved}", flush=True)
        else:
            print(f"[resume] 체크포인트 요청 경로: {resolved}", flush=True)
        if model_pt is None:
            print(
                "[resume] 이전 정책을 찾을 수 없습니다. 처음부터 학습을 시작합니다.",
                flush=True,
            )
        else:
            print(
                f"[resume] 로드할 모델 파일: {model_pt.expanduser().resolve()}",
                flush=True,
            )


def _model_checkpoint_sort_key(p: Path) -> int:
    if not p.stem.startswith("model_"):
        return -1
    try:
        return int(p.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _iter_numeric_model_pts(dir_path: Path) -> list[Path]:
    return [
        p
        for p in dir_path.glob("model_*.pt")
        if _model_checkpoint_sort_key(p) >= 0
    ]


def _pick_latest_checkpoint_by_mtime(candidates: list[Path]) -> Path | None:
    """가장 최근 저장(mtime). 동시각이면 파일명 스텝이 큰 쪽."""
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda p: (p.stat().st_mtime, _model_checkpoint_sort_key(p)),
    )


def resolve_resume_checkpoint(path: Path | None) -> tuple[Path | None, Path | None]:
    """Return (model_*.pt path, trainer_state.pt path or None)."""
    if path is None:
        return None, None
    path = path.expanduser()
    if not path.exists():
        return None, None
    if path.is_file():
        if path.suffix != ".pt":
            return None, None
        trainer = path.parent / "trainer_state.pt"
        return path, trainer if trainer.exists() else None
    models = _iter_numeric_model_pts(path)
    if not models:
        return None, None
    model = _pick_latest_checkpoint_by_mtime(models)
    trainer = path / "trainer_state.pt"
    return model, trainer if trainer.exists() else None


def load_policy_checkpoint(policy: nn.Module, path: Path, device: str) -> None:
    resolved = path.expanduser().resolve()
    obj = torch.load(resolved, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        policy.load_state_dict(obj.state_dict(), strict=False)
    elif isinstance(obj, dict) and "state_dict" in obj:
        policy.load_state_dict(obj["state_dict"], strict=False)
    else:
        policy.load_state_dict(obj, strict=False)


class Vectorization(Enum):
    multiprocessing = "multiprocessing"
    serial = "serial"
    ray = "ray"


def make_policy(env: RedGymEnv, policy_name: str, config: DictConfig) -> nn.Module:
    validate_config_choice(config, "policies", policy_name)
    policy_module_name, policy_class_name = policy_name.split(".")
    policy_module = importlib.import_module(f"pokemonred_puffer.policies.{policy_module_name}")
    policy_class = getattr(policy_module, policy_class_name)

    policy = policy_class(env, **config.policies[policy_name].policy)
    if config.train.use_rnn:
        rnn_config = config.policies[policy_name].rnn
        policy_class = getattr(policy_module, rnn_config.name)
        policy = policy_class(env, policy, **rnn_config.args)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(config.train.device)


def load_from_config(config: DictConfig, debug: bool) -> DictConfig:
    default_keys = ["env", "train", "policies", "rewards", "wrappers", "wandb"]
    defaults = OmegaConf.create({key: config.get(key, {}) for key in default_keys})

    # Package and subpackage (environment) configs
    debug_config = config.get("debug", OmegaConf.create({})) if debug else OmegaConf.create({})

    defaults.merge_with(debug_config)
    return defaults


def validate_config_choice(config: DictConfig, section: str, choice: str) -> None:
    available = config.get(section)
    if available is None or choice not in available:
        options = [] if available is None else sorted(list(available.keys()))
        raise KeyError(
            f"Unknown {section[:-1]} '{choice}'. Available {section}: {options}"
        )


def make_env_creator(
    wrapper_classes: list[tuple[str, ModuleType]],
    reward_class: RedGymEnv,
    async_wrapper: bool = False,
    sqlite_wrapper: bool = False,
    puffer_wrapper: bool = True,
) -> Callable[[DictConfig, DictConfig], pufferlib.emulation.GymnasiumPufferEnv | gymnasium.Env]:
    def env_creator(
        env_config: DictConfig,
        wrappers_config: list[dict[str, Any]],
        reward_config: DictConfig,
        async_config: dict[str, Queue] | None = None,
        sqlite_config: dict[str, str] | None = None,
    ) -> pufferlib.emulation.GymnasiumPufferEnv | gymnasium.Env:
        env = reward_class(env_config, reward_config)
        for cfg, (_, wrapper_class) in zip(wrappers_config, wrapper_classes):
            env = wrapper_class(env, OmegaConf.create([x for x in cfg.values()][0]))
        if async_wrapper and async_config:
            env = AsyncWrapper(env, async_config["send_queues"], async_config["recv_queues"])
        if sqlite_wrapper and sqlite_config:
            env = SqliteStateResetWrapper(env, sqlite_config["database"])
        if puffer_wrapper:
            env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
        return env

    return env_creator


def setup_agent(
    wrappers: list[str],
    reward_name: str,
    async_wrapper: bool = False,
    sqlite_wrapper: bool = False,
    puffer_wrapper: bool = True,
) -> Callable[[DictConfig, DictConfig], pufferlib.emulation.GymnasiumPufferEnv]:
    # TODO: Make this less dependent on the name of this repo and its file structure
    wrapper_classes = [
        (
            k,
            getattr(
                importlib.import_module(f"pokemonred_puffer.wrappers.{k.split('.')[0]}"),
                k.split(".")[1],
            ),
        )
        for wrapper_dicts in wrappers
        for k in wrapper_dicts.keys()
    ]
    reward_module, reward_class_name = reward_name.split(".")
    reward_class = getattr(
        importlib.import_module(f"pokemonred_puffer.rewards.{reward_module}"), reward_class_name
    )
    # NOTE: This assumes reward_module has RewardWrapper(RedGymEnv) class
    env_creator = make_env_creator(
        wrapper_classes, reward_class, async_wrapper, sqlite_wrapper, puffer_wrapper
    )

    return env_creator


@contextmanager
def init_wandb(
    config: DictConfig,
    exp_name: str,
    reward_name: str,
    policy_name: str,
    wrappers_name: str,
    resume: bool = True,
):
    if not config.track:
        yield None
    else:
        assert config.wandb.project is not None, "Please set the wandb project in config.yaml"
        assert config.wandb.entity is not None, "Please set the wandb entity in config.yaml"
        wandb_kwargs = {
            "id": exp_name or wandb.util.generate_id(),
            "project": config.wandb.project,
            "entity": config.wandb.entity,
            "group": config.wandb.group,
            "config": {
                "cleanrl": config.train,
                "env": config.env,
                "reward_module": reward_name,
                "policy_module": policy_name,
                "reward": config.rewards[reward_name],
                "policy": config.policies[policy_name],
                "wrappers": config.wrappers[wrappers_name],
                "rnn": "rnn" in config.policies[policy_name],
            },
            "name": exp_name,
            "monitor_gym": True,
            "save_code": True,
            "resume": resume,
        }
        client = wandb.init(**wandb_kwargs)
        yield client
        client.finish()


def setup(
    config: DictConfig,
    debug: bool,
    wrappers_name: str,
    reward_name: str,
    rom_path: Path,
    track: bool,
    puffer_wrapper: bool = True,
) -> tuple[DictConfig, Callable[[DictConfig, DictConfig], pufferlib.emulation.GymnasiumPufferEnv]]:
    validate_config_choice(config, "wrappers", wrappers_name)
    validate_config_choice(config, "rewards", reward_name)

    # Allow user to fix exp_id in config; otherwise, generate one.
    exp_id = str(getattr(config.train, "exp_id", "")).strip()
    if not exp_id:
        config.train.exp_id = f"pokemon-red-{str(uuid.uuid4())[:8]}"
    config.env.gb_path = rom_path
    config.track = track
    if debug:
        config.vectorization = Vectorization.serial

    async_wrapper = config.train.get("async_wrapper", False)
    sqlite_wrapper = config.train.get("sqlite_wrapper", False)
    env_creator = setup_agent(
        config.wrappers[wrappers_name], reward_name, async_wrapper, sqlite_wrapper, puffer_wrapper
    )
    return config, env_creator


@app.command()
def evaluate(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    checkpoint_path: Path | None = None,
    policy_name: Annotated[
        str,
        typer.Option(
            "--policy-name",
            "-p",
            help="Policy module to use in policies.",
        ),
    ] = DEFAULT_POLICY,
    reward_name: Annotated[
        str,
        typer.Option(
            "--reward-name",
            "-r",
            help="Reward module to use in rewards",
        ),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option(
            "--wrappers-name",
            "-w",
            help="Wrappers to use _in order of instantion_",
        ),
    ] = DEFAULT_WRAPPER,
    rom_path: Path = DEFAULT_ROM,
):
    config, env_creator = setup(
        config=config,
        debug=False,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=False,
    )
    env_kwargs = {
        "env_config": config.env,
        "wrappers_config": config.wrappers[wrappers_name],
        "reward_config": config.rewards[reward_name]["reward"],
        "async_config": {},
    }
    try:
        cleanrl_puffer.rollout(
            env_creator,
            env_kwargs,
            model_path=checkpoint_path,
            device=config.train.device,
        )
    except KeyboardInterrupt:
        os._exit(0)


@app.command()
def autotune(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    policy_name: Annotated[
        str,
        typer.Option(
            "--policy-name",
            "-p",
            help="Policy module to use in policies.",
        ),
    ] = DEFAULT_POLICY,
    reward_name: Annotated[
        str,
        typer.Option(
            "--reward-name",
            "-r",
            help="Reward module to use in rewards",
        ),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option(
            "--wrappers-name",
            "-w",
            help="Wrappers to use _in order of instantion_",
        ),
    ] = DEFAULT_WRAPPER,
    rom_path: Path = DEFAULT_ROM,
):
    config = load_from_config(config, False)
    config.vectorization = "multiprocessing"
    config, env_creator = setup(
        config=config,
        debug=False,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=False,
    )
    env_kwargs = {
        "env_config": config.env,
        "wrappers_config": config.wrappers[wrappers_name],
        "reward_config": config.rewards[reward_name]["reward"],
        "async_config": {},
    }
    pufferlib.vector.autotune(
        functools.partial(env_creator, **env_kwargs), batch_size=config.train.env_batch_size
    )


def _debug_format_exploration_reward_lines(info: dict[str, Any]) -> list[str]:
    """탐험 등: reward/* + reward_sum 을 한 블록으로."""
    import numpy as np

    keys = sorted(k for k in info if k.startswith("reward/") or k == "reward_sum")
    if not keys:
        return []
    parts: list[str] = []
    for k in keys:
        v = info[k]
        short = k.replace("reward/", "", 1) if k.startswith("reward/") else k
        if isinstance(v, np.ndarray):
            parts.append(f"{short}=ndarray{v.shape}")
        elif isinstance(v, (bool, np.bool_)):
            parts.append(f"{short}={v}")
        elif isinstance(v, (float, int, np.floating, np.integer)):
            parts.append(f"{short}={float(v):.6g}")
        else:
            parts.append(f"{short}={v!r}")
    # 한 줄이 너무 길면 두 줄로
    line = "  └─ reward   " + "  ".join(parts)
    if len(line) <= 100:
        return [line]
    mid = max(1, len(parts) // 2)
    return [
        "  └─ reward   " + "  ".join(parts[:mid]),
        "              " + "  ".join(parts[mid:]),
    ]


def _debug_format_game_context_line(info: dict[str, Any]) -> list[str]:
    """배틀/맵/마지막 행동만 (선택 키가 있을 때만)."""
    import numpy as np

    keys = ("stats/in_battle", "stats/last_action", "stats/map_id", "stats/step")
    parts: list[str] = []
    for k in keys:
        if k not in info:
            continue
        v = info[k]
        short = k.replace("stats/", "", 1)
        if isinstance(v, np.ndarray):
            parts.append(f"{short}=ndarray{v.shape}")
        elif isinstance(v, (bool, np.bool_)):
            parts.append(f"{short}={v}")
        elif isinstance(v, (float, int, np.floating, np.integer)):
            parts.append(f"{short}={v}")
        else:
            parts.append(f"{short}={v!r}")
    if not parts:
        return []
    return ["  └─ game     " + "  ".join(parts)]


@app.command()
def debug(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    reward_name: Annotated[
        str,
        typer.Option(
            "--reward-name",
            "-r",
            help="Reward module to use in rewards",
        ),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option(
            "--wrappers-name",
            "-w",
            help="Wrappers to use _in order of instantion_",
        ),
    ] = DEFAULT_WRAPPER,
    rom_path: Path = DEFAULT_ROM,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="info 전체 출력 (터미널 폭주 주의)"),
    ] = False,
    print_every: Annotated[
        int,
        typer.Option(
            "--print-every",
            "-n",
            help="N번 step 루프마다 한 번 로그 출력 (에피소드 종료 시에는 항상 출력)",
        ),
    ] = 10,
    step_delay: Annotated[
        float,
        typer.Option(
            "--step-delay",
            "-s",
            help="루프 한 번당 대기(초). 로그 속도와 별개로 에뮬 루프 간격.",
        ),
    ] = 0.5,
):
    config = load_from_config(config, True)
    config.env.gb_path = rom_path
    config, env_creator = setup(
        config=config,
        debug=True,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=False,
        puffer_wrapper=False,
    )
    env = env_creator(
        config.env, config.wrappers[wrappers_name], config.rewards[reward_name]["reward"]
    )
    obs, reset_info = env.reset()
    print("[debug] reset 완료 — step 루프 시작 (Ctrl+C로 종료)", flush=True)
    if reset_info:
        print(f"[debug] reset_info keys: {list(reset_info.keys())}", flush=True)

    def _debug_print_step(
        loop_i: int, reward: float, terminated: bool, truncated: bool, info: dict[str, Any]
    ) -> None:
        base = env.unwrapped
        step_count = getattr(base, "step_count", None)
        head = (
            f"[debug] iter={loop_i}  env_step={step_count}  "
            f"r_step={reward:+.6f}  terminated={terminated}  truncated={truncated}"
        )
        if info:
            er = info.get("episode_return")
            el = info.get("episode_length")
            if er is not None:
                head += f"  episode_return={float(er):+.6f}"
            if el is not None:
                head += f"  episode_length={el}"
        print(head, flush=True)
        if not info:
            return
        if verbose:
            extra = {k: v for k, v in info.items() if k not in ("episode_return", "episode_length", "state")}
            if extra:
                print(f"  └─ info_full  {extra}", flush=True)
            return
        for line in _debug_format_exploration_reward_lines(info):
            print(line, flush=True)
        for line in _debug_format_game_context_line(info):
            print(line, flush=True)

    pe = max(1, print_every)
    try:
        loop_i = 0
        while True:
            loop_i += 1
            obs, reward, terminated, truncated, info = env.step(5)
            if verbose or loop_i == 1 or (loop_i % pe == 0) or terminated or truncated:
                _debug_print_step(loop_i, reward, terminated, truncated, info)
            if terminated or truncated:
                obs, info = env.reset()
                print("[debug] 에피소드 리셋", flush=True)
            time.sleep(max(0.0, step_delay))
    finally:
        env.close()


@app.command()
def train(
    config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = DEFAULT_CONFIG,
    policy_name: Annotated[
        str,
        typer.Option(
            "--policy-name",
            "-p",
            help="Policy module to use in policies.",
        ),
    ] = DEFAULT_POLICY,
    reward_name: Annotated[
        str,
        typer.Option(
            "--reward-name",
            "-r",
            help="Reward module to use in rewards",
        ),
    ] = DEFAULT_REWARD,
    wrappers_name: Annotated[
        str,
        typer.Option(
            "--wrappers-name",
            "-w",
            help="Wrappers to use _in order of instantion_",
        ),
    ] = DEFAULT_WRAPPER,
    exp_name: Annotated[str | None, typer.Option(help="Resume from experiment")] = None,
    checkpoint_path: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint-path",
            "-c",
            help="이전 학습 디렉터리(runs/pokemon-red-*) 또는 model_*.pt; config의 train.resume_checkpoint를 덮어씀",
        ),
    ] = None,
    resume_latest: Annotated[
        bool,
        typer.Option(
            "--resume-latest",
            help="명시적 경로 없이 최신 체크포인트 탐색(기본 동작과 동일, 호환용)",
        ),
    ] = False,
    fresh: Annotated[
        bool,
        typer.Option(
            "--fresh",
            help="체크포인트 자동 불러오기 끄고 가중치 없이 처음부터 학습",
        ),
    ] = False,
    rom_path: Path = DEFAULT_ROM,
    track: Annotated[bool, typer.Option(help="Track on wandb.")] = False,
    debug: Annotated[bool, typer.Option(help="debug")] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Rich 터미널 대시보드(스텝/SPS/손실/환경 스칼라 표)",
        ),
    ] = False,
    vectorization: Annotated[
        Vectorization, typer.Option(help="Vectorization method")
    ] = "multiprocessing",
):
    config = load_from_config(config, debug)
    if verbose:
        config.train.verbose = True
    config.vectorization = vectorization
    config, env_creator = setup(
        config=config,
        debug=debug,
        wrappers_name=wrappers_name,
        reward_name=reward_name,
        rom_path=rom_path,
        track=track,
    )
    with init_wandb(
        config=config,
        exp_name=exp_name,
        reward_name=reward_name,
        policy_name=policy_name,
        wrappers_name=wrappers_name,
    ) as wandb_client:
        vec = config.vectorization
        if vec == Vectorization.serial:
            vec = pufferlib.vector.Serial
        elif vec == Vectorization.multiprocessing:
            vec = pufferlib.vector.Multiprocessing
        elif vec == Vectorization.ray:
            vec = pufferlib.vector.Ray
        else:
            vec = pufferlib.vector.Multiprocessing

        resume_src, resume_auto_attempted, resume_fresh = _effective_resume_path(
            checkpoint_path, config, resume_latest_cli=resume_latest, fresh_cli=fresh
        )
        model_pt, trainer_pt = resolve_resume_checkpoint(resume_src)
        _log_resume_intent(
            resume_src, resume_auto_attempted, model_pt, resume_fresh=resume_fresh
        )

        # TODO: Remove the +1 once the driver env doesn't permanently increase the env id
        env_send_queues = []
        env_recv_queues = []
        if config.train.get("async_wrapper", False):
            env_send_queues = [Queue() for _ in range(2 * config.train.num_envs + 1)]
            env_recv_queues = [Queue() for _ in range(2 * config.train.num_envs + 1)]

        sqlite_context = nullcontext
        if config.train.get("sqlite_wrapper", False):
            sqlite_context = functools.partial(NamedTemporaryFile, suffix="sqlite")

        with sqlite_context() as sqlite_db:
            db_filename = None
            if config.train.get("sqlite_wrapper", False):
                db_filename = sqlite_db.name
                with sqlite3.connect(db_filename) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS states(env_id INTEGER PRIMARY KEY, pyboy_state BLOB, reset BOOLEAN, required_rate REAL, pid INT);"
                    )

            vecenv = pufferlib.vector.make(
                env_creator,
                env_kwargs={
                    "env_config": config.env,
                    "wrappers_config": config.wrappers[wrappers_name],
                    "reward_config": config.rewards[reward_name]["reward"],
                    "async_config": {
                        "send_queues": env_send_queues,
                        "recv_queues": env_recv_queues,
                    },
                    "sqlite_config": {"database": db_filename},
                },
                num_envs=config.train.num_envs,
                num_workers=config.train.num_workers,
                batch_size=config.train.env_batch_size,
                zero_copy=config.train.zero_copy,
                backend=vec,
            )
            policy = make_policy(vecenv.driver_env, policy_name, config)

            resume_trainer_state = None
            resume_load_log_lines: list[str] | None = None
            if model_pt is not None:
                load_policy_checkpoint(
                    policy, model_pt, str(config.train.device)
                )
                resolved_model = model_pt.expanduser().resolve()
                lines = [f"[resume] 정책 가중치 로드 완료: {resolved_model}"]
                if trainer_pt is not None and config.train.get(
                    "load_optimizer_state", False
                ):
                    trainer_resolved = trainer_pt.expanduser().resolve()
                    resume_trainer_state = torch.load(
                        trainer_pt,
                        map_location=str(config.train.device),
                        weights_only=False,
                    )
                    lines.append(
                        f"[resume] 옵티마이저 상태 로드 완료: {trainer_resolved}"
                    )
                resume_load_log_lines = lines

            config.train.env = "Pokemon Red"
            with CleanPuffeRL(
                exp_name=exp_name,
                config=config.train,
                vecenv=vecenv,
                policy=policy,
                env_recv_queues=env_recv_queues,
                env_send_queues=env_send_queues,
                sqlite_db=db_filename,
                wandb_client=wandb_client,
                resume_trainer_state=resume_trainer_state,
                resume_load_log_lines=resume_load_log_lines,
            ) as trainer:
                while not trainer.done_training():
                    trainer.evaluate()
                    trainer.train()

            print("Done training")


if __name__ == "__main__":
    app()
