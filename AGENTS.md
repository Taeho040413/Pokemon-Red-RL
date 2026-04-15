# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Pokemon Red RL — a reinforcement learning agent that plays Pokemon Red via the PyBoy Game Boy emulator, trained using PufferLib (CleanRL-based PPO). Single-machine Python project; no Docker, databases, or web servers required.

### Python version

The project requires **Python >= 3.10, < 3.12**. A Python 3.10 virtual environment is set up at `/workspace/.venv`. Always activate it before running commands:

```sh
source /workspace/.venv/bin/activate
```

### Key commands

See `README.md` and `pyproject.toml` for full details. Quick reference:

| Task | Command |
|---|---|
| Install (dev) | `pip install -e '.[dev]'` |
| Lint check | `ruff check pokemonred_puffer` |
| Format check | `ruff format --check pokemonred_puffer` |
| Tests | `python -m pytest tests` |
| CLI help | `python3 -m pokemonred_puffer.train --help` |
| Train | `python3 -m pokemonred_puffer.train train` |
| Debug mode | `python3 -m pokemonred_puffer.train --config config.yaml --debug` |

### Running the application end-to-end

Training and evaluation require one external asset **not included in the repository** (copyrighted / user-supplied):

- **Pokemon Red ROM** (`red.gb`) — must be placed at the workspace root. The file is gitignored (`*.gb` in `.gitignore`) so it cannot be committed. Without this file, `train`, `debug`, and `evaluate` commands will fail with `FileNotFoundError: ROM file red.gb was not found!`.

The **PyBoy save states** (`pyboy_states/*.state`) are already committed. The default init state is `pyboy_states/PewterCity.state` (configured in `config.yaml` → `env.state_dir` + `env.init_state`).

Tests (`python -m pytest tests`) use mocked PyBoy and do **not** require the ROM or save state.

### Hello world verification (without ROM)

Even without the ROM, you can verify the dev environment works by:
1. Running tests: `python -m pytest tests -v` (6 tests, all pass)
2. CLI check: `python3 -m pokemonred_puffer.train --help`
3. Environment instantiation with mocked PyBoy (tests do this automatically)

### Non-obvious caveats

- The Cython extension `c_gae.pyx` is auto-compiled on first import via `pyximport`. A C compiler (`gcc`) must be available.
- `opencv-python==3.4.17.63` is pinned to an old version; do not upgrade it without testing.
- `pufferlib` is installed from a custom fork (`git+https://github.com/thatguy11325/PufferLib.git@1.0`), not PyPI.
- `ruff format --check` currently reports 8 files that would be reformatted (pre-existing in the repo). `ruff check` passes cleanly.
- The `gym` deprecation warning on import is expected; the project uses both `gym` (via PufferLib) and `gymnasium`.
