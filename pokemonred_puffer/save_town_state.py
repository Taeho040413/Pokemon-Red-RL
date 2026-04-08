import argparse
import io
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from pyboy import PyBoy
from pyboy.utils import WindowEvent

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROM_PATH = _PROJECT_ROOT / "pokered.gb"
DEFAULT_RAM_PATH = _PROJECT_ROOT / "pokered.gb.ram"
STATE_DIR = _PROJECT_ROOT / "pyboy_states"
POKERED_SYM = _PROJECT_ROOT / "pokemonred_puffer" / "pokered.sym"


def export_sav_to_state(
    sav_path: Path,
    out_state_path: Path,
    *,
    rom_path: Path = DEFAULT_ROM_PATH,
    warmup_ticks: int = 2400,
    headless: bool = True,
) -> Path:
    """배터리 `.sav`를 PyBoy `.state` 한 파일로 변환 (학습용 고정 시작점)."""
    if not sav_path.is_file():
        raise FileNotFoundError(f"SAV not found: {sav_path}")
    if not rom_path.is_file():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    ram = io.BytesIO(sav_path_to_ram_bytes(sav_path))
    sym = str(POKERED_SYM) if POKERED_SYM.is_file() else None
    pyboy = PyBoy(
        str(rom_path),
        ram_file=ram,
        window="null" if headless else "SDL2",
        log_level="CRITICAL",
        symbols=sym,
        sound_emulated=False,
    )
    # From SRAM, game still needs title/menu inputs before reaching overworld.
    # Reuse the same start/A pattern as interactive helper.
    auto_skip_intro(pyboy, max_ticks=max(1, warmup_ticks))
    out_state_path = out_state_path.resolve()
    out_state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_state_path, "wb") as f:
        pyboy.save_state(f)
    pyboy.stop(save=False)
    print(f"Exported: {out_state_path} ({out_state_path.stat().st_size} bytes)", flush=True)
    return out_state_path


def stdin_worker(cmd_q: Queue):
    print("명령 형식: save 이름   예) save ViridianCity", flush=True)
    print("종료 형식: q", flush=True)
    while True:
        try:
            cmd = input("> ").strip()
        except EOFError:
            cmd = "q"
        cmd_q.put(cmd)
        if cmd.lower() == "q":
            break


def sav_path_to_ram_bytes(sav_path: Path) -> bytes:
    """Extract 32KB cartridge SRAM from a battery `.sav` for use with PyBoy ``ram_file=``."""
    data = sav_path.read_bytes()
    size = len(data)
    if size < 32768:
        raise ValueError(f"SAV too small: {sav_path} ({size} bytes)")
    if size == 32768:
        return data
    if size == 32772:
        return data[:32768]
    return data[:32768]


def install_ram_from_sav(sav_path: Path, ram_path: Path):
    data = sav_path.read_bytes()
    size = len(data)
    if size < 32768:
        raise ValueError(f"SAV too small: {sav_path} ({size} bytes)")

    if size == 32768:
        strategy = "raw-32KB"
    elif size == 32772:
        strategy = "32772->head32KB"
    else:
        head_candidate = sav_path.with_suffix(".head32k.ram")
        tail_candidate = sav_path.with_suffix(".tail32k.ram")
        head_candidate.write_bytes(data[:32768])
        tail_candidate.write_bytes(data[-32768:])
        strategy = "fallback-head32KB"
        print(
            f"주의: 비표준 SAV 크기({size}). 후보 생성: {head_candidate}, {tail_candidate}",
            flush=True,
        )

    out = sav_path_to_ram_bytes(sav_path)
    backup_path = ram_path.with_suffix(ram_path.suffix + ".bak")
    if ram_path.exists() and not backup_path.exists():
        backup_path.write_bytes(ram_path.read_bytes())
        print(f"기존 RAM 백업: {backup_path}", flush=True)

    ram_path.write_bytes(out)
    print(f"SAV 적용 완료 ({strategy}): {ram_path} ({len(out)} bytes)", flush=True)


def auto_skip_intro(pyboy: PyBoy, max_ticks: int = 2400):
    # Try to pass title/intro and land on overworld quickly.
    for i in range(max_ticks):
        if i % 24 == 0:
            pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        if i % 24 == 1:
            pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        if i % 12 == 0:
            pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        if i % 12 == 1:
            pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        if pyboy.tick() is False:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rom-path",
        type=Path,
        default=DEFAULT_ROM_PATH,
        help="ROM path to run in PyBoy",
    )
    parser.add_argument(
        "--ram-path",
        type=Path,
        default=DEFAULT_RAM_PATH,
        help="RAM path used by PyBoy battery save",
    )
    parser.add_argument(
        "--sav-path",
        type=Path,
        default=None,
        help="Optional .sav path to convert/apply into --ram-path before launching",
    )
    parser.add_argument(
        "--auto-skip-intro",
        action="store_true",
        help="Auto send START/A inputs to skip intro/title and continue",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="PyBoy emulation speed multiplier (e.g. 3 for 3x)",
    )
    parser.add_argument(
        "--export-sav",
        type=Path,
        default=None,
        help="배터리 .sav 경로 — 지정 시 비대화형으로 .state만 만들고 종료",
    )
    parser.add_argument(
        "--export-state-out",
        type=Path,
        default=None,
        help="--export-sav 와 함께: 저장할 .state 경로 (예: pyboy_states/PewterCity.state)",
    )
    parser.add_argument(
        "--warmup-ticks",
        type=int,
        default=2400,
        help="export 시 자동 START/A 입력 tick 수 (기본 2400)",
    )
    args = parser.parse_args()

    if args.export_sav is not None:
        if args.export_state_out is None:
            raise SystemExit("--export-sav 를 쓰면 --export-state-out 도 필수입니다.")
        export_sav_to_state(
            args.export_sav.expanduser().resolve(),
            args.export_state_out.expanduser().resolve(),
            rom_path=args.rom_path.expanduser().resolve(),
            warmup_ticks=args.warmup_ticks,
            headless=True,
        )
        return

    rom_path = args.rom_path
    ram_path = args.ram_path
    print(f"rom exists: {rom_path.exists()} -> {rom_path}", flush=True)
    print(f"ram exists: {ram_path.exists()} -> {ram_path}", flush=True)

    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")

    if args.sav_path is not None:
        if not args.sav_path.exists():
            raise FileNotFoundError(f"SAV not found: {args.sav_path}")
        ram_path.parent.mkdir(parents=True, exist_ok=True)
        install_ram_from_sav(args.sav_path, ram_path)
    elif not ram_path.exists():
        raise FileNotFoundError(f"RAM not found: {ram_path}")

    STATE_DIR.mkdir(exist_ok=True)

    pyboy = PyBoy(str(rom_path), window="SDL2")
    if args.speed < 1:
        raise ValueError("--speed must be >= 1")
    pyboy.set_emulation_speed(args.speed)
    print("PyBoy 실행됨. SDL 창에서 Continue로 들어가서 공중날기 쓰면 됨.", flush=True)
    if args.auto_skip_intro:
        print("인트로/타이틀 자동 스킵 시도 중...", flush=True)
        auto_skip_intro(pyboy)
        print("자동 스킵 입력 완료. 화면 확인 후 save 명령 입력하세요.", flush=True)

    cmd_q = Queue()
    t = Thread(target=stdin_worker, args=(cmd_q,), daemon=True)
    t.start()

    running = True
    while running:
        # 게임을 실제로 진행시키는 핵심
        still_running = pyboy.tick()
        if still_running is False:
            break

        try:
            cmd = cmd_q.get_nowait()
        except Empty:
            continue

        if not cmd:
            continue

        if cmd.lower() == "q":
            running = False
            continue

        if cmd.lower().startswith("save "):
            name = cmd[5:].strip()
            if not name:
                print("이름이 비었음. 예: save PalletTown", flush=True)
                continue

            out_path = STATE_DIR / f"{name}.state"
            with open(out_path, "wb") as f:
                pyboy.save_state(f)
            print(f"저장 완료: {out_path}", flush=True)
        else:
            print("알 수 없는 명령. 예: save ViridianCity / q", flush=True)

    pyboy.stop()
    print("종료됨", flush=True)

if __name__ == "__main__":
    main()