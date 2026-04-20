import numpy as np
from omegaconf import DictConfig, OmegaConf
from pyboy.utils import WindowEvent

from pokemonred_puffer.data.events import EVENTS
from pokemonred_puffer.data.items import MAX_ITEM_CAPACITY
from pokemonred_puffer.data.map import MapIds
from pokemonred_puffer.data.tilesets import Tilesets
from pokemonred_puffer.environment import VALID_ACTIONS, RedGymEnv
from pokemonred_puffer.global_map import local_to_global


# 오픈 필드(마을·도로·사천): 여기서 나가 처음 들어가는 구역 = new_building 후보
_FIELD_TILESETS: frozenset[int] = frozenset(
    {
        Tilesets.OVERWORLD.value,
        Tilesets.PLATEAU.value,
    }
)
# 숲·게이트 등: 필드↔연결 구간은 new_building, 연결↔연결(다른 맵)은 new_room
_CONNECTOR_TILESETS: frozenset[int] = frozenset(
    {
        Tilesets.GATE.value,
        Tilesets.FOREST_GATE.value,
        Tilesets.FOREST.value,
    }
)

# pokered: FIRST_INDOOR_MAP == REDS_HOUSE_1F (0x25) — 그 미만은 마을·도로 등 야외 맵 ID
_LAST_OUTDOOR_MAP_ID: int = MapIds.ROUTE_25.value

# GATE / FOREST / FOREST_GATE 타일셋을 쓰는 맵 (타일셋이 0으로 읽힐 때 보조 분류)
_CONNECTOR_MAP_IDS: frozenset[int] = frozenset(
    m.value
    for m in (
        MapIds.VIRIDIAN_FOREST_NORTH_GATE,
        MapIds.ROUTE_2_GATE,
        MapIds.VIRIDIAN_FOREST_SOUTH_GATE,
        MapIds.VIRIDIAN_FOREST,
        MapIds.ROUTE_5_GATE,
        MapIds.ROUTE_6_GATE,
        MapIds.ROUTE_7_GATE,
        MapIds.ROUTE_8_GATE,
        MapIds.ROUTE_11_GATE_1F,
        MapIds.ROUTE_11_GATE_2F,
        MapIds.ROUTE_12_GATE_1F,
        MapIds.ROUTE_12_GATE_2F,
        MapIds.SAFARI_ZONE_GATE,
        MapIds.ROUTE_15_GATE_1F,
        MapIds.ROUTE_15_GATE_2F,
        MapIds.ROUTE_16_GATE_1F,
        MapIds.ROUTE_16_GATE_2F,
        MapIds.ROUTE_18_GATE_1F,
        MapIds.ROUTE_18_GATE_2F,
        MapIds.ROUTE_22_GATE,
    )
)

_POKECENTER_MAP_IDS: frozenset[int] = frozenset(
    {
        MapIds.VIRIDIAN_POKECENTER.value,
        MapIds.PEWTER_POKECENTER.value,
        MapIds.CERULEAN_POKECENTER.value,
        MapIds.MT_MOON_POKECENTER.value,
        MapIds.ROCK_TUNNEL_POKECENTER.value,
        MapIds.VERMILION_POKECENTER.value,
        MapIds.CELADON_POKECENTER.value,
        MapIds.LAVENDER_POKECENTER.value,
        MapIds.FUCHSIA_POKECENTER.value,
        MapIds.CINNABAR_POKECENTER.value,
        MapIds.SAFFRON_POKECENTER.value,
    }
)


class BaselineRewardEnv(RedGymEnv):
    def __init__(self, env_config: DictConfig, reward_config: DictConfig):
        super().__init__(env_config)
        self.reward_config = OmegaConf.to_object(reward_config)
        self.max_event_rew = 0
        self.max_level_sum = 0

    def get_game_state_reward(self):
        raise NotImplementedError(
            "Use ExplorationInteractionRewardEnv instead of BaselineRewardEnv."
        )

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def get_all_events_reward(self):
        return max(
            np.sum(self.events.get_events(EVENTS))
            - self.base_event_flags
            - int(self.events.get_event("EVENT_BOUGHT_MUSEUM_TICKET")),
            0,
        )

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        return 15 + (self.max_level_sum - 15) / 4


class ExplorationInteractionRewardEnv(BaselineRewardEnv):
    def init_mem(self):
        super().init_mem()
        self._reset_interaction_tracking()

    def reset_mem(self):
        super().reset_mem()
        self._reset_interaction_tracking()

    def _reset_interaction_tracking(self):
        self.item_count = 0
        self.gym_core_npc_count = 0
        self.first_npc_talk_count = 0
        self.first_object_interaction_count = 0
        self.new_tile_count = 0
        self.new_building_count = 0
        self.new_room_count = 0
        self.new_npc_textbox_count = 0
        self.repeat_npc_interaction_count = 0
        self.invalid_interaction_count = 0
        self.start_menu_open_count = 0
        self.stuck_penalty_count = 0
        # 야생전 진입(wIsInBattle: 0 -> 1) 1회당 패널티
        self.wild_encounter_count = 0
        # 파티 슬롯별 HP가 >0 → 0(기절) 1회당, 야생전에서만 누적
        self.death_count = 0
        self.trainer_battle_win_count = 0
        self.pokecenter_first_entry_count = 0
        self.pokecenter_heal_hp_count = 0
        self._same_coord_streak = 0
        self._last_coord_for_stuck: tuple[int, int, int] | None = None

        self._seen_object_ids: set[tuple[str, int, int]] = set()
        self._seen_unique_coords: set[tuple[int, int, int]] = set()
        self._seen_building_map_ids: set[int] = set()
        self._seen_pokecenter_entries: set[int] = set()
        self._seen_room_map_ids: set[int] = set()
        self._seen_npc_textboxes: set[tuple[int, int]] = set()

        self._last_blackout_map_id: int | None = None
        self._last_bag_item_counts: dict[int, int] = {}
        self._item_kinds_ever_obtained: set[int] = set()
        self._pending_npc_key: tuple[int, int] | None = None

        self._interaction_triggered_this_step = False
        self._reward_state_seeded = False
        # 전멸 후 센터 워프·부활 구간에서는 pokecenter/new_building 셰이핑으로 꿀빨지 않게 함
        self._suppress_pokecenter_shaping_after_blackout = False

    def _reward(self, key: str) -> float:
        return float(self.reward_config.get(key, 0.0))

    def _textbox_active(self) -> bool:
        return bool(self.read_m("wTextBoxID") != 0 or self.read_m("wFontLoaded") != 0)

    def _read_party_hp_sum(self) -> int:
        """Total current HP across the entire party."""
        party_size = int(self.read_m("wPartyCount"))
        return int(sum(self.read_short(f"wPartyMon{i+1}HP") for i in range(party_size)))

    @staticmethod
    def _tileset_kind(tileset: int) -> str:
        """field | connector | interior — 건물/방 구분에 사용."""
        if tileset in _FIELD_TILESETS:
            return "field"
        if tileset in _CONNECTOR_TILESETS:
            return "connector"
        return "interior"

    @classmethod
    def _tileset_kind_for_structure(cls, map_id: int, raw_tileset: int) -> str:
        """맵 전환 직후 wCurMapTileset==0 인 프레임 보정 + 맵 ID 기반 보조 분류."""
        if raw_tileset != 0:
            return cls._tileset_kind(raw_tileset)
        if map_id <= _LAST_OUTDOOR_MAP_ID:
            return "field"
        if map_id in _CONNECTOR_MAP_IDS:
            return "connector"
        return "interior"

    @staticmethod
    def _is_pokecenter_map(map_id: int) -> bool:
        return map_id in _POKECENTER_MAP_IDS

    def _register_new_building(self, map_id: int) -> None:
        if map_id in self._seen_building_map_ids:
            return
        self._seen_building_map_ids.add(map_id)
        if self._is_pokecenter_map(map_id) and self._suppress_pokecenter_shaping_after_blackout:
            return
        self.new_building_count += 1

    def _register_new_room(self, map_id: int) -> None:
        if map_id in self._seen_room_map_ids:
            return
        self._seen_room_map_ids.add(map_id)
        self.new_room_count += 1

    def _maybe_reward_pokecenter_entry(self, map_id: int) -> None:
        if not self._is_pokecenter_map(map_id) or map_id in self._seen_pokecenter_entries:
            return
        if self._suppress_pokecenter_shaping_after_blackout:
            return
        self._seen_pokecenter_entries.add(map_id)
        self.pokecenter_first_entry_count += 1

    def _try_register_structure_visit(self, map_id: int, *, as_building: bool) -> None:
        """맵 ID당 최초 1회만 new_building 또는 new_room 카운트."""
        if map_id in self._seen_building_map_ids or map_id in self._seen_room_map_ids:
            return
        if as_building:
            self._register_new_building(map_id)
            self._maybe_reward_pokecenter_entry(map_id)
        else:
            self._register_new_room(map_id)

    def _mark_building_entry_tile(self, map_id: int) -> None:
        """현재 위치를 건물 진입 타일(보라색 오버레이)로 기록."""
        x_pos, y_pos, cur_map_id = self.get_game_coords()
        if int(cur_map_id) != int(map_id):
            return
        gy, gx = local_to_global(y_pos, x_pos, int(map_id))
        self.building_entry_tile_map[gy, gx] = 1.0

    def _apply_map_change_structure_reward(
        self,
        prev_map_id: int,
        prev_tileset: int,
        map_id: int,
        cur_tileset: int,
    ) -> None:
        """wCurMap 이 바뀐 한 번에 대해 new_building / new_room 분기.

        반드시 에뮬레이터 한 액션의 (시작 시점 맵, 종료 시점 맵) 비교에서만 호출한다.
        update_seen_coords(첫 tick 직후)와 Joy 루프 이후를 따로 보면 같은 워프에 대해 서로 다른
        prev_tileset 이 잡혀 new_room / new_building 이 뒤틀릴 수 있다.

        new_building으로 가야 할 전환(야외/게이트→실내)이 new_room으로 가면 안 되므로,
        pokered 상 맵 ID가 FIRST_INDOOR_MAP(0x25) 미만이면 **항상 field**로 본다
        (타일셋 0·한 프레임 오독으로 interior로 분류되는 경우 차단).
        """
        if self._is_pokecenter_map(int(prev_map_id)) and not self._is_pokecenter_map(int(map_id)):
            self._suppress_pokecenter_shaping_after_blackout = False

        prev_kind = self._tileset_kind_for_structure(prev_map_id, prev_tileset)
        if prev_map_id <= _LAST_OUTDOOR_MAP_ID:
            prev_kind = "field"

        cur_kind = self._tileset_kind_for_structure(map_id, cur_tileset)
        if map_id <= _LAST_OUTDOOR_MAP_ID:
            cur_kind = "field"

        if self._is_pokecenter_map(map_id):
            self._mark_building_entry_tile(map_id)
            self._try_register_structure_visit(map_id, as_building=True)
            return

        if cur_kind == "field":
            return

        if cur_kind == "interior":
            as_building = prev_kind in ("field", "connector")
            if prev_kind == "interior":
                as_building = False
            if as_building:
                self._mark_building_entry_tile(map_id)
            self._try_register_structure_visit(map_id, as_building=as_building)
            return

        # cur_kind == "connector" (숲·게이트 등 다른 맵으로 이동)
        if prev_kind == "field":
            self._mark_building_entry_tile(map_id)
            self._try_register_structure_visit(map_id, as_building=True)
        elif prev_kind == "connector":
            self._try_register_structure_visit(map_id, as_building=False)

    def _ensure_pokecenter_entry_recorded(self, map_id: int, cur_tileset: int) -> None:
        if not self._is_pokecenter_map(map_id):
            return

        # If the doorway transition was skipped by map loading timing, heal can be
        # observed before the first-entry bookkeeping. Backfill the center entry,
        # and only synthesize new_building when this indoor map has never been seen.
        if (
            self._tileset_kind_for_structure(map_id, cur_tileset) != "field"
            and map_id not in self._seen_building_map_ids
            and map_id not in self._seen_room_map_ids
        ):
            self._register_new_building(map_id)

        self._maybe_reward_pokecenter_entry(map_id)

    def _get_bag_item_counts(self) -> dict[int, int]:
        # wNumBagItems is a uint8 in WRAM. If it reads as 0 (or wraps unexpectedly),
        # the slice addr:addr+0 becomes invalid for PyBoy's memory view.
        num_bag_items_raw = self.read_m("wNumBagItems")
        num_bag_items = int(num_bag_items_raw)
        _, addr = self.pyboy.symbol_lookup("wBagItems")

        start_addr = int(addr)
        end_addr = start_addr + 2 * num_bag_items
        if num_bag_items <= 0 or end_addr <= start_addr:
            return {}

        # Clamp in case of any unexpected wrap.
        num_bag_items = min(num_bag_items, MAX_ITEM_CAPACITY)
        raw = self.pyboy.memory[start_addr : start_addr + 2 * num_bag_items]
        return {
            int(raw[i]): int(raw[i + 1])
            for i in range(0, len(raw), 2)
            if int(raw[i]) != 0 and int(raw[i]) != 0xFF
        }

    def _seed_reward_state_if_needed(self):
        if self._reward_state_seeded:
            return

        x_pos, y_pos, map_id = self.get_game_coords()
        cur_tileset = self.read_m("wCurMapTileset")
        self._seen_unique_coords.add((x_pos, y_pos, map_id))

        # 필드가 아닌 곳에서 시작: 해당 맵은 이미 "방문한 구역"으로 두어 첫 전환 보상이 꼬이지 않게 함
        if self._tileset_kind_for_structure(map_id, cur_tileset) != "field":
            self._seen_room_map_ids.add(map_id)

        self._last_blackout_map_id = int(self.read_m("wLastBlackoutMap"))
        self._last_bag_item_counts = self._get_bag_item_counts()
        self._item_kinds_ever_obtained = {
            item_id
            for item_id, cnt in self._last_bag_item_counts.items()
            if cnt > 0
        }
        self._reward_state_seeded = True

    def _update_bag_item_tracking(self):
        current_counts = self._get_bag_item_counts()
        for item_id, count in current_counts.items():
            if count <= 0 or item_id in self._item_kinds_ever_obtained:
                continue
            self._item_kinds_ever_obtained.add(item_id)
            self.item_count += 1

        self._last_bag_item_counts = current_counts

    def update_seen_coords(self):
        self._seed_reward_state_if_needed()

        super().update_seen_coords()

        x_pos, y_pos, map_id = self.get_game_coords()
        cur_coord = (x_pos, y_pos, map_id)

        if cur_coord not in self._seen_unique_coords:
            self._seen_unique_coords.add(cur_coord)
            self.new_tile_count += 1

        # Penalize prolonged no-movement loops only outside battle/textbox.
        if cur_coord == self._last_coord_for_stuck:
            self._same_coord_streak += 1
        else:
            self._same_coord_streak = 1
            self._last_coord_for_stuck = cur_coord

        if (
            self.read_m("wIsInBattle") == 0
            and not self._textbox_active()
            and self._same_coord_streak > 10
        ):
            # stuck 페널티는 카운트로만 처리한다(빨간색 타일맵은 야생 조우 시각화로 재배정).
            self.stuck_penalty_count += 1

    def run_action_on_emulator(self, action):
        self._seed_reward_state_if_needed()
        # 구조 보상: 이 액션 직전 WRAM vs Joy 루프까지 포함한 직후 WRAM (한 스텝·한 번만 판정)
        map_before = int(self.read_m("wCurMap"))
        ts_before = int(self.read_m("wCurMapTileset"))
        self._interaction_triggered_this_step = False
        pressed_a = VALID_ACTIONS[action] == WindowEvent.PRESS_BUTTON_A

        hp_sum_before = int(self._read_party_hp_sum())
        prev_pokecenter_heal = int(self.pokecenter_heal)
        prev_blackout_count = int(self.blackout_count)
        prev_is_in_battle = int(self.read_m("wIsInBattle"))
        party_n = max(0, min(int(self.read_m("wPartyCount")), 6))
        hp_before_slots = [
            int(self.read_short(f"wPartyMon{i+1}HP")) for i in range(party_n)
        ]

        super().run_action_on_emulator(action)

        map_after = int(self.read_m("wCurMap"))
        ts_after = int(self.read_m("wCurMapTileset"))
        if self._reward_state_seeded and map_before != map_after:
            self._apply_map_change_structure_reward(map_before, ts_before, map_after, ts_after)

        current_blackout_map_id = int(self.read_m("wLastBlackoutMap"))
        self._last_blackout_map_id = current_blackout_map_id

        if int(self.blackout_count) > prev_blackout_count:
            self._suppress_pokecenter_shaping_after_blackout = True

        # 개체 기절(슬롯 HP >0 → 0): 야생전(wIsInBattle==1)만 death_count. 트레이너전(2)은 무패널티.
        # 마지막 기절 후 배틀 플래그가 이미 0이면 prev_is_in_battle으로 복원.
        post_battle = int(self.read_m("wIsInBattle"))
        battle_ctx = post_battle if post_battle in (1, 2) else prev_is_in_battle
        party_n2 = max(0, min(int(self.read_m("wPartyCount")), 6))
        for i in range(min(party_n, party_n2)):
            hp_after_slot = int(self.read_short(f"wPartyMon{i+1}HP"))
            if hp_before_slots[i] > 0 and hp_after_slot == 0 and battle_ctx == 1:
                self.death_count += 1

        # 트레이너전 승리만 보상 (wIsInBattle 2→0, 동일 스텝 블랙아웃 없음). 야생 승리/도주는 제외.
        if (
            prev_is_in_battle == 2
            and post_battle == 0
            and int(self.blackout_count) == prev_blackout_count
        ):
            self.trainer_battle_win_count += 1

        # 야생 조우: 필드(0) → 야생(1)만. 트레이너는 2라서 제외.
        if prev_is_in_battle == 0 and post_battle == 1:
            self.wild_encounter_count += 1
            # 조우 발생 좌표를 Kanto 오버레이에 빨간 강도로 누적 기록
            enc_x, enc_y, enc_map = self.get_game_coords()
            _gy, _gx = local_to_global(enc_y, enc_x, enc_map)
            if 0 <= _gy < self.wild_encounter_tile_map.shape[0] and (
                0 <= _gx < self.wild_encounter_tile_map.shape[1]
            ):
                self.wild_encounter_tile_map[_gy, _gx] = min(
                    self.wild_encounter_tile_map[_gy, _gx] + 1.0, 1e4
                )

        # Pokémon Center healing reward:
        # - pokecenter_heal is set by AnimateHealingMachine hook
        # - reward proportional to total party HP gained
        # - exclude revival-from-blackout (hp_sum_before == 0)
        did_blackout = int(self.blackout_count) > prev_blackout_count
        if self.pokecenter_heal == 1 and prev_pokecenter_heal == 0:
            hp_sum_after = int(self._read_party_hp_sum())
            if (
                hp_sum_before > 0
                and not did_blackout
                and not self._suppress_pokecenter_shaping_after_blackout
            ):
                healed = max(0, hp_sum_after - hp_sum_before)
                current_map_id = int(self.read_m("wCurMap"))
                current_tileset = int(self.read_m("wCurMapTileset"))
                self._ensure_pokecenter_entry_recorded(current_map_id, current_tileset)
                self.pokecenter_heal_hp_count += healed

        # One-shot so we don't double count across steps.
        if int(self.pokecenter_heal) == 1:
            self.pokecenter_heal = 0

        self._update_bag_item_tracking()
        self._update_script_and_text_tracking()

        if (
            pressed_a
            and not self._interaction_triggered_this_step
            and self.read_m("wIsInBattle") == 0
            and not self._textbox_active()
        ):
            self.invalid_interaction_count += 1

    def start_menu_hook(self, *args, **kwargs):
        super().start_menu_hook(*args, **kwargs)
        if self.read_m("wIsInBattle") == 0:
            self.start_menu_open_count += 1

    def sign_hook(self, *args, **kwargs):
        sign_id = self.read_m("hSpriteIndexOrTextID")
        map_id = self.read_m("wCurMap")
        self.seen_signs[(map_id, sign_id)] = 1.0
        self._interaction_triggered_this_step = True

        object_key = ("sign", map_id, sign_id)
        if object_key not in self._seen_object_ids:
            self._seen_object_ids.add(object_key)
            self.first_object_interaction_count += 1

    def hidden_object_hook(self, *args, **kwargs):
        _, addr = self.pyboy.symbol_lookup("wHiddenObjectIndex")
        hidden_object_id = int(self.pyboy.memory[addr])
        map_id = self.read_m("wCurMap")
        self.seen_hidden_objs[(map_id, hidden_object_id)] = 1.0
        self._interaction_triggered_this_step = True

        object_key = ("hidden", map_id, hidden_object_id)
        if object_key not in self._seen_object_ids:
            self._seen_object_ids.add(object_key)
            self.first_object_interaction_count += 1

    def sprite_hook(self, *args, **kwargs):
        sprite_id = self.read_m("hSpriteIndexOrTextID")
        map_id = self.read_m("wCurMap")
        npc_key = (map_id, sprite_id)
        was_seen = npc_key in self.seen_npcs

        self.seen_npcs[npc_key] = 1.0
        self._interaction_triggered_this_step = True
        self._pending_npc_key = npc_key

        if was_seen:
            self.repeat_npc_interaction_count += 1
        else:
            self.first_npc_talk_count += 1
            # Proxy for "gym core npc": first interaction with a gym NPC.
            if self.read_m("wCurMapTileset") == Tilesets.GYM.value:
                self.gym_core_npc_count += 1

    def _update_script_and_text_tracking(self):
        if self._pending_npc_key is not None and self._textbox_active():
            if self._pending_npc_key not in self._seen_npc_textboxes:
                self._seen_npc_textboxes.add(self._pending_npc_key)
                self.new_npc_textbox_count += 1

        if not self._textbox_active():
            self._pending_npc_key = None

    def get_game_state_reward(self) -> dict[str, float]:
        self._seed_reward_state_if_needed()

        return {
            "event": self._reward("event") * self.update_max_event_rew(),
            # item_count = 에피소드 동안 처음 가방에 들어온 아이템 종류 수 (시작 가방 제외)
            "item": self._reward("item") * self.item_count,
            "gym_core_npc": self._reward("gym_core_npc") * self.gym_core_npc_count,
            "npc_first_talk": self._reward("npc_first_talk") * self.first_npc_talk_count,
            "object_first_interaction": self._reward("object_first_interaction")
            * self.first_object_interaction_count,
            "new_tile": self._reward("new_tile") * self.new_tile_count,
            "new_building": self._reward("new_building") * self.new_building_count,
            "new_room": self._reward("new_room") * self.new_room_count,
            "new_npc_textbox": self._reward("new_npc_textbox") * self.new_npc_textbox_count,
            "step_penalty": self._reward("step_penalty") * self.step_count,
            "repeat_npc_penalty": self._reward("repeat_npc_penalty")
            * self.repeat_npc_interaction_count,
            "invalid_interaction": self._reward("invalid_interaction")
            * self.invalid_interaction_count,
            "start_menu_penalty": self._reward("start_menu_penalty")
            * self.start_menu_open_count,
            "stuck_penalty": self._reward("stuck_penalty") * self.stuck_penalty_count,
            "wild_encounter_penalty": self._reward("wild_encounter_penalty")
            * self.wild_encounter_count,
            "trainer_battle_win": self._reward("trainer_battle_win")
            * self.trainer_battle_win_count,
            "death": self._reward("death") * self.death_count,
            "pokecenter_first_entry": self._reward("pokecenter_first_entry")
            * self.pokecenter_first_entry_count,
            "pokecenter_heal_hp": self._reward("pokecenter_heal_hp")
            * self.pokecenter_heal_hp_count,
        }
