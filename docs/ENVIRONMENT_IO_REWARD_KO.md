# 포켓몬 레드 RL 환경 — 입출력·액션·리워드·패널티·구조

이 문서는 `pokemonred_puffer.environment.RedGymEnv` 및 `pokemonred_puffer.rewards.baseline.ExplorationInteractionRewardEnv`(기본 학습 설정)를 기준으로 정리합니다. 기본 `config.yaml`의 `env.reduce_res: True`, `env.two_bit: True`, `env.use_global_map: False`를 가정합니다.

---

## 0. Pokemon Red RL (PufferLib) — 구조 개요

환경 → 정책 → 보상의 데이터 흐름을 한 화면에 모은 개요입니다. (**현재 레포 코드** 기준.)

<table>
<tr valign="top">
<td width="33%">

### 환경 — PyBoy / `RedGymEnv`

**관측 (`Dict`)**

- **시각·공간**
  - `screen` — 화면 버퍼 (2비트 패킹 가능)
  - `visited_mask` — 방문 오버레이
  - `global_map` — *(선택)* `env.use_global_map=True`일 때만 공간에 추가
- **플레이어·맵 상태**
  - `direction`, `map_id`, `blackout_map_id`, `battle_type`
- **인벤·파티**
  - `bag_items` / `bag_quantity`
  - 파티 6슬롯: `species`, `hp`, `maxHP`, `status`, `type1`, `type2`, `level`, `attack`, `defense`, `speed`, `special`, `moves`
- **진행도**
  - `events` — 이벤트 플래그 비트열 (정책에서 일부 인덱스만 사용)

**보조 정보 (관측 아님 · `info` / 로그)**

- 미사블 등 예: `game_corner_rocket`
- 사파리: `info["stats"]["safari_zone"]` 등

**출력**

- **액션**: 이산 **7**개 (`↓←→↑`, `A`, `B`, `START`)

</td>
<td width="33%">

### 정책 — `MultiConvolutionalPolicy` (+ RNN)

학습 설정에서 `MultiConvolutionalRNN`(LSTM)이 정책을 감쌉니다 (`config.yaml` → `use_rnn`).

**입력 처리**

1. **CNN** (`screen` ∥ `visited_mask` 채널 결합) → 화면 특징 벡터
2. **임베딩·표 처리**
   - `map_id`, `blackout_map_id` → `nn.Embedding`
   - 가방: 아이템 ID 임베딩 × 수량 스케일
   - 파티: 종족·타입·기술 임베딩 + 스탯 정규화 → `party_network`
   - `direction`, `battle_type` → one-hot
   - `events` → 비트 언팩 후 `EVENTS_IDXS`만 사용
3. *(선택)* `global_map` → 별도 CNN 분기 후 벡터 결합

**은닉·헤드**

- 전부 연결 → `encode_linear` → 은닉 **`z`**
- **Value**: `value_fn` → **V(s)** (스칼라)
- **Actor**: `actor` → **7 logits** (위 액션과 1:1)

**비고**

- 별도 **HM 서브헤드**(cut/surf 등 확률)는 **없음**. 필드 기술은 환경 옵션 `auto_teach_*` / `auto_use_*`로 에뮬레이터가 처리할 수 있음.

</td>
<td width="33%">

### 보상 — `ExplorationInteractionRewardEnv`

**부모**: `BaselineRewardEnv` → `RedGymEnv`

**누적 딕셔너리**

- `get_game_state_reward()`가 항목별 가중 합을 반환 (예: `event`, `new_tile`, `new_building`, `wild_encounter_penalty`, …).

**스텝 보상**

- `update_reward()`:  
  **`step_reward = sum(새 딕셔너리) − sum(이전 누적)`**  
  각 항목은 에피소드 동안 단조 증가하는 카운트 × `config` 계수.

**셰이핑 요약**

- **긍정**: 이벤트, 신규 타일·건물·방, NPC/오브젝트 최초 상호작용, 트레이너 승리, 포켓센터 등
- **패널티**: 스텝, 정체(`stuck`), 야생 조우, 반복 NPC, 무효 A, 시작 메뉴, 야생전 기절 등

**가중치**

- `config.yaml` → `rewards.baseline.ExplorationInteractionRewardEnv.reward`

**비고**

- **Reward Machine**(상태 `rm_*`) 전이 가중은 **본 레포 보상 클래스에는 없음**.

</td>
</tr>
</table>

### 한 줄 흐름

```mermaid
flowchart LR
  subgraph Env["환경\nPyBoy / RedGymEnv"]
    O["관측 Dict\n(screen, visited_mask, …)"]
    A["액션 ×7"]
  end
  subgraph Pol["정책\nMultiConvolutionalPolicy + RNN"]
    E["CNN · 임베딩 · concat"]
    Z["은닉 z"]
    V["V(s)"]
    Pi["π: 7 logits"]
  end
  subgraph Rew["보상\nExplorationInteractionRewardEnv"]
    D["항목별 누적"]
    S["스텝 보상 = Δ합"]
  end
  O --> E --> Z
  Z --> V
  Z --> Pi
  Pi --> A
  A --> Env
  Env --> D --> S
```

---

## 1. 입출력 (I/O)

### 1.1 관측 (`observation`)

Gymnasium `Dict` 공간. 에이전트는 매 스텝 딕셔너리 관측을 받습니다.

| 키 | 의미 | 형태 (기본 설정) | 비고 |
|----|------|------------------|------|
| `screen` | 게임 화면 (그레이스케일/압축) | `(72, 20, 1)` uint8 | `reduce_res`·`two_bit` 시 가로 80→20 패킹 |
| `visited_mask` | 방문 마스크 오버레이 | `screen`과 동일 shape | 탐험 시각화용 |
| `direction` | 플레이어 방향 | `(1,)` 0~4 | |
| `blackout_map_id` | 마지막 블랙아웃 시 맵 ID | `(1,)` | WRAM `wLastBlackoutMap` 등 |
| `battle_type` | 전투 종류 | `(1,)` 0~2 | 0=비전투, 1=야생, 2=트레이너 |
| `map_id` | 현재 맵 ID | `(1,)` | |
| `bag_items` | 가방 아이템 ID (슬롯) | `(20,)` | 빈 슬롯은 0 등 |
| `bag_quantity` | 수량 | `(20,)` | |
| `species` | 파티 종족값 | `(6,)` | |
| `hp` / `maxHP` | HP | `(6,)` uint32 | |
| `status` | 상태 이상 | `(6,)` | |
| `type1` / `type2` | 타입 | `(6,)` | |
| `level` | 레벨 | `(6,)` | |
| `attack` / `defense` / `speed` / `special` | 스탯 | `(6,)` | |
| `moves` | 기술 (슬롯 4) | `(6, 4)` | |
| `events` | 이벤트 플래그 비트열 | `(320,)` 0~1 | |

`env.use_global_map: True`이면 `global_map` 채널이 추가됩니다 (전역 탐험 맵).

### 1.2 스텝 반환값 (`step`)

표준 Gymnasium API: `(obs, reward, terminated, truncated, info)`.

- **`reward`**: 아래 [리워드 합](#3-리워드-cumulative--incremental)의 **스텝 증분** (`update_reward`).
- **`terminated` (`reset`)**: 에피소드 종료 플래그. 예: 스타팅으로 컷 불가능 파티, 관장 첫 배지 획득(`end_episode_on_first_gym`), 필수 진행률 허용 오차 초과(`required_tolerance`) 등.
- **`info`**: 주기적으로 `agent_stats` 등 (설정에 따라 `state` 바이트 등).

---

## 2. 액션 (Action space)

`gymnasium.spaces.Discrete(7)` — 인덱스 `0`~`6`.

| 인덱스 | 이름 (`VALID_ACTIONS_STR`) | PyBoy 입력 |
|--------|----------------------------|------------|
| 0 | `down` | 아래 화살표 press |
| 1 | `left` | 왼쪽 |
| 2 | `right` | 오른쪽 |
| 3 | `up` | 위 |
| 4 | `a` | A |
| 5 | `b` | B |
| 6 | `start` | START |

한 스텝당 에뮬레이터는 해당 키를 눌렀다 떼고, `action_freq`틱만큼 진행합니다 (`run_action_on_emulator`).

---

## 3. 리워드 (cumulative → incremental)

`ExplorationInteractionRewardEnv.get_game_state_reward()`는 **항목별 누적 기여도** `{키: 값}` 딕셔너리를 반환합니다. 스칼라 보상은 `update_reward()`에서

**이번 스텝 보상 = (새로운 전체 합) − (직전 전체 합)**

으로 계산됩니다. 따라서 각 항목은 “카운트 × 계수” 형태로 **에피소드 동안 단조 증가(또는 감소)하는 누적량**에 가중치를 곱한 값입니다.

### 3.1 긍정적 셰이핑 (리워드 계수 > 0)

`config.yaml`의 `rewards.baseline.ExplorationInteractionRewardEnv.reward` 기본값 예시:

| 키 | 의미 (요약) | 기본 계수 |
|----|----------------|-----------|
| `event` | 스토리 이벤트 플래그 진행 | × `5.0` |
| `item` | 에피소드 중 **처음** 가방에 들어온 아이템 **종류** 수 | × `2.5` |
| `gym_core_npc` | 처음 대화한 체육관 NPC (타일셋 GYM) | × `1.5` |
| `npc_first_talk` | NPC 최초 대화 | × `3.0` |
| `object_first_interaction` | 간판/히든오브젝트 등 최초 상호작용 | × `0.6` |
| `new_tile` | **새 좌표** 방문 (풀숲 야생 타일 제외) | × `0.014` |
| `new_building` | **건물/구역** 최초 진입 (필드·게이트→실내 등 규칙) | × `4.0` |
| `new_room` | 연결 구역 간 **새 방/맵** (커넥터 규칙) | × `2.0` |
| `new_npc_textbox` | NPC 대화창 최초 오픈 | × `0.6` |
| `trainer_battle_win` | **트레이너전** 승리 (야생 제외) | × `0.15` |
| `pokecenter_first_entry` | 포켓몬센터 최초 입장 | × `4.0` |
| `pokecenter_heal_hp` | 센터 회복량 (전멸 부활 HP 급등 등은 제외 로직) | × `0.005` (HP 단위) |

### 3.2 패널티·음수 셰이핑 (계수 ≤ 0)

| 키 | 의미 (요약) | 기본 계수 |
|----|----------------|-----------|
| `step_penalty` | 스텝 수에 비례 | × `-0.00004` per step |
| `repeat_npc_penalty` | 같은 NPC 반복 대화 | × `-0.07` |
| `invalid_interaction` | 전투/텍스트 없이 빈 A 입력 등 | × `-0.002` |
| `start_menu_penalty` | 시작 메뉴 오픈 | × `-0.022` |
| `stuck_penalty` | 같은 칸 10프레임 초과 정체 (전투·텍스트 제외) | × `-0.004` |
| `wild_encounter_penalty` | 야생 조우 발생 1회당 | × `-1.2` |
| `death` | 야생전에서 슬롯 HP가 >0→0 (기절) | × `-0.5` |

에피소드 초기화 시 특수 처리: 스타팅에 **컷 가능 포켓몬이 없으면** `reward`에 `-0.5 * total_reward` 페널티가 가해지고 리셋됩니다 (`step` 내부).

---

## 4. “구조” 보상 — 건물 vs 방 (new_building / new_room)

맵 전환 시 `wCurMap`, `wCurMapTileset`으로 **필드(field) / 커넥터(숲·게이트) / 실내(interior)** 를 구분합니다.

- **야외 맵 ID** (`≤ ROUTE_25`)는 타일셋이 잠깐 0이어도 항상 `field`로 취급해 오분류를 막습니다.
- **포켓몬 센터**는 별도 규칙·전멸 후 억제 플래그(`_suppress_pokecenter_shaping_after_blackout`)로 과도한 셰이핑을 줄입니다.
- 필드·커넥터에서 실내로 들어가면 `new_building`, 커넥터↔다른 구역 등은 `new_room` 등으로 카운트합니다.

아래는 한 스텝 안에서의 판정 흐름 요약입니다.

```mermaid
flowchart TD
  A[액션 실행 전: map_before, tileset_before] --> B[에뮬레이터 Joy 루프 및 틱]
  B --> C[액션 실행 후: map_after, tileset_after]
  C --> D{map 변경?}
  D -->|아니오| Z[구조 카운트 변화 없음]
  D -->|예| E[이전/현재 타일셋 종류: field / connector / interior]
  E --> F{포켓몬 센터 맵?}
  F -->|예| G[new_building 등 센터 규칙 + 입구 타일 마킹]
  F -->|아니오| H{현재 kind}
  H -->|field| Z
  H -->|interior| I{이전이 field 또는 connector?}
  I -->|예| J[new_building 후보 + 입구 타일]
  I -->|아니오| K[new_room 후보]
  H -->|connector| L{이전 kind}
  L -->|field| J
  L -->|connector| K
```

---

## 5. 설정 변경 시 주의

- `reward.*` 계수는 `config.yaml`의 `rewards.baseline.ExplorationInteractionRewardEnv.reward`에서 바뀝니다.
- 관측 shape은 `reduce_res`, `two_bit`, `use_global_map`에 따라 달라지므로 정책 입력 차원과 일치시켜야 합니다.
