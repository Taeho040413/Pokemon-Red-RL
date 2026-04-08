import numpy as np
from omegaconf import DictConfig, OmegaConf
from pyboy.utils import WindowEvent

from pokemonred_puffer.data.events import EVENTS
from pokemonred_puffer.data.items import MAX_ITEM_CAPACITY
from pokemonred_puffer.data.tilesets import Tilesets
from pokemonred_puffer.environment import VALID_ACTIONS, RedGymEnv
from pokemonred_puffer.global_map import local_to_global

# "Outside" for reward split: towns/routes/plateau/connectors/forest — not buildings, caves, gyms, etc.
# new_building: surface → non-surface; new_room: non-surface → non-surface (different map).
_OUTDOOR_SURFACE_TILESETS: frozenset[int] = frozenset(
    {
        Tilesets.OVERWORLD.value,
        Tilesets.PLATEAU.value,
        Tilesets.GATE.value,
        Tilesets.FOREST_GATE.value,
        Tilesets.FOREST.value,
    }
)

# pokered ram/wram.asm: $00 win, $01 lose, $02 draw
_BATTLE_RESULT_WIN = 0


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
    """Exploration + interaction reward env.

    Intended reward keys:
    - event
    - item
    - gym_core_npc
    - npc_first_talk
    - object_first_interaction
    - new_tile
    - new_building (outdoor surface tileset → indoor; see _OUTDOOR_SURFACE_TILESETS)
    - new_room (indoor → indoor, different map)
    - new_npc_textbox
    - script_step
    - step_penalty
    - repeat_npc_penalty
    - invalid_interaction
    - start_menu_penalty
    - stuck_penalty
    - battle_win (temporary; wBattleResult win after leaving battle)
    - pokecenter_heal_hp (temporary; sum(HP gained) at Pokémon Center, excluding revival-from-blackout)
    """

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
        self.script_step_count = 0
        self.repeat_npc_interaction_count = 0
        self.invalid_interaction_count = 0
        self.start_menu_open_count = 0
        self.stuck_penalty_count = 0
        self.battle_win_count = 0
        self.pokecenter_heal_hp_count = 0
        self._prev_is_in_battle = 0
        self._same_coord_streak = 0
        self._last_coord_for_stuck: tuple[int, int, int] | None = None

        self._seen_object_ids: set[tuple[str, int, int]] = set()
        self._seen_unique_coords: set[tuple[int, int, int]] = set()
        self._seen_building_map_ids: set[int] = set()
        self._seen_room_map_ids: set[int] = set()
        self._seen_npc_textboxes: set[tuple[int, int]] = set()

        self._last_map_id: int | None = None
        self._last_tileset: int | None = None
        self._last_script_state: tuple[int, int, int] | None = None
        self._last_bag_item_counts: dict[int, int] = {}
        self._pending_npc_key: tuple[int, int] | None = None

        self._interaction_triggered_this_step = False
        self._reward_state_seeded = False
        self._prev_is_in_battle = int(self.read_m("wIsInBattle"))

    def _reward(self, key: str) -> float:
        return float(self.reward_config.get(key, 0.0))

    def _textbox_active(self) -> bool:
        return bool(self.read_m("wTextBoxID") != 0 or self.read_m("wFontLoaded") != 0)

    def _read_party_hp_sum(self) -> int:
        """Total current HP across the entire party."""
        party_size = int(self.read_m("wPartyCount"))
        return int(sum(self.read_short(f"wPartyMon{i+1}HP") for i in range(party_size)))

    @staticmethod
    def _is_outdoor_surface_tileset(tileset: int) -> bool:
        return tileset in _OUTDOOR_SURFACE_TILESETS

    def _current_script_state(self) -> tuple[int, int, int]:
        return (
            self.read_m("wCurMap"),
            self.read_m("wCurMapScript"),
            self.read_short("wCurMapScriptPtr"),
        )

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

        # Start indoors (cave/mart/gym/…): seed room set so indoor↔indoor is new_room only.
        # Outdoor surface starts (route/town/plateau/forest) do not seed — first door gives new_building.
        if not self._is_outdoor_surface_tileset(cur_tileset):
            self._seen_room_map_ids.add(map_id)

        self._last_map_id = map_id
        self._last_tileset = cur_tileset
        self._last_script_state = self._current_script_state()
        self._last_bag_item_counts = self._get_bag_item_counts()
        self._reward_state_seeded = True

    def _update_bag_item_tracking(self):
        current_counts = self._get_bag_item_counts()
        for item_id, count in current_counts.items():
            prev_count = self._last_bag_item_counts.get(item_id, 0)
            if count > prev_count:
                self.item_count += count - prev_count

        self._last_bag_item_counts = current_counts

    def update_seen_coords(self):
        self._seed_reward_state_if_needed()
        prev_map_id = self._last_map_id
        prev_tileset = self._last_tileset

        super().update_seen_coords()

        x_pos, y_pos, map_id = self.get_game_coords()
        cur_tileset = self.read_m("wCurMapTileset")
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
            self.stuck_penalty_count += 1
            _gy, _gx = local_to_global(y_pos, x_pos, map_id)
            self.stuck_tile_map[_gy, _gx] = min(self.stuck_tile_map[_gy, _gx] + 1.0, 1e4)

        if prev_map_id is not None and map_id != prev_map_id:
            prev_out = self._is_outdoor_surface_tileset(prev_tileset)
            cur_out = self._is_outdoor_surface_tileset(cur_tileset)
            # Exterior → interior (town/route/plateau/… → mart/gym/cave/…)
            if prev_out and not cur_out:
                if map_id not in self._seen_building_map_ids:
                    self._seen_building_map_ids.add(map_id)
                    self.new_building_count += 1
            # Interior → interior (stairs, another house floor, etc.) — mutually exclusive with new_building
            elif not prev_out and not cur_out:
                if map_id not in self._seen_room_map_ids:
                    self._seen_room_map_ids.add(map_id)
                    self.new_room_count += 1

        self._last_map_id = map_id
        self._last_tileset = cur_tileset

    def run_action_on_emulator(self, action):
        self._seed_reward_state_if_needed()
        self._interaction_triggered_this_step = False
        pressed_a = VALID_ACTIONS[action] == WindowEvent.PRESS_BUTTON_A

        hp_sum_before = int(self._read_party_hp_sum())
        prev_pokecenter_heal = int(self.pokecenter_heal)
        prev_blackout_count = int(self.blackout_count)

        prev_ib = self._prev_is_in_battle
        super().run_action_on_emulator(action)
        cur_ib = int(self.read_m("wIsInBattle"))
        if prev_ib != 0 and cur_ib == 0:
            if int(self.read_m("wBattleResult")) == _BATTLE_RESULT_WIN:
                self.battle_win_count += 1
        self._prev_is_in_battle = cur_ib

        # Pokémon Center healing reward:
        # - pokecenter_heal is set by AnimateHealingMachine hook
        # - reward proportional to total party HP gained
        # - exclude revival-from-blackout (hp_sum_before == 0)
        did_blackout = int(self.blackout_count) > prev_blackout_count
        if self.pokecenter_heal == 1 and prev_pokecenter_heal == 0:
            hp_sum_after = int(self._read_party_hp_sum())
            if hp_sum_before > 0 and not did_blackout:
                healed = max(0, hp_sum_after - hp_sum_before)
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

        current_script_state = self._current_script_state()
        if self._last_script_state is None:
            self._last_script_state = current_script_state
        elif (
            current_script_state != self._last_script_state
            and self._pending_npc_key is not None
            and self._textbox_active()
        ):
            self.script_step_count += 1
            self._last_script_state = current_script_state
        else:
            self._last_script_state = current_script_state

        if not self._textbox_active():
            self._pending_npc_key = None

    def get_game_state_reward(self) -> dict[str, float]:
        self._seed_reward_state_if_needed()

        return {
            "event": self._reward("event") * self.update_max_event_rew(),
            "item": self._reward("item") * self.item_count,
            "gym_core_npc": self._reward("gym_core_npc") * self.gym_core_npc_count,
            "npc_first_talk": self._reward("npc_first_talk") * self.first_npc_talk_count,
            "object_first_interaction": self._reward("object_first_interaction")
            * self.first_object_interaction_count,
            "new_tile": self._reward("new_tile") * self.new_tile_count,
            "new_building": self._reward("new_building") * self.new_building_count,
            "new_room": self._reward("new_room") * self.new_room_count,
            "new_npc_textbox": self._reward("new_npc_textbox") * self.new_npc_textbox_count,
            "script_step": self._reward("script_step") * self.script_step_count,
            "step_penalty": self._reward("step_penalty") * self.step_count,
            "repeat_npc_penalty": self._reward("repeat_npc_penalty")
            * self.repeat_npc_interaction_count,
            "invalid_interaction": self._reward("invalid_interaction")
            * self.invalid_interaction_count,
            "start_menu_penalty": self._reward("start_menu_penalty")
            * self.start_menu_open_count,
            "stuck_penalty": self._reward("stuck_penalty") * self.stuck_penalty_count,
            "battle_win": self._reward("battle_win") * self.battle_win_count,
            "pokecenter_heal_hp": self._reward("pokecenter_heal_hp")
            * self.pokecenter_heal_hp_count,
        }
