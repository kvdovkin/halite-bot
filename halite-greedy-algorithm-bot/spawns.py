import numpy as np

from convert import working_yards
from settings import (SPAWNING_STEP, STEPS_FINAL, MIN_SHIPS, SPAWNING_OFFSET,
                      YARD_SCHEDULE)


class Spawns:
    def __init__(self, state, actions):
        # определить, сколько кораблей построить
        self.num_ships(state, actions)

        # позиции верфей, для которых мы еще можем принять решение
        yards = [state.my_yards[yard] for yard in actions.yards]
        self.spawn_pos = np.array(yards, dtype=int)

        # сортировать позиции верфей по предпочтению для спауна - спаун
        # где мало наших кораблей не предпочтителен
        inds = np.ix_(state.my_ship_pos, self.spawn_pos)
        traffic = np.sum(state.dist[inds] <= 3, axis=0, initial=0)
        not_working = ~np.in1d(self.spawn_pos, working_yards(state))
        score = traffic + 10 * not_working.astype(int)
        self.spawn_pos = self.spawn_pos[score.argsort()]
        return

    def num_ships(self, state, actions):
        # определяем, сколько кораблей мы хотели бы создать на основе всех игроков
        # количество кораблей и оценка = галит + груз
        ships = state.my_ship_pos.size
        score = state.my_halite + np.sum(state.my_ship_hal)
        max_opp_ships = max(state.opp_num_ships.values(), default=0)
        max_opp_score = max(state.opp_scores.values(), default=0)

        # оставить не менее MIN_SHIPS кораблей
        bound = MIN_SHIPS - ships
        new_ships = max(bound, 0)

        # появляется, если у нас меньше кораблей, чем у противников
        # но относимся к этому менее серьезно в конце игры
        offset = SPAWNING_OFFSET * (state.step / state.total_steps)
        bound = max_opp_ships - offset - ships
        new_ships = max(bound, new_ships)

        # появляется, если у нас больше халита, чем противников. буфер ниже
        # ~ 500 (= spawnCost) в середине игры и ~ 1500
        # (= spawnCost + 2 * convertCost) когда осталось <50 шагов
        op_costs = state.spawn_cost + state.convert_cost
        buffer = 2 * op_costs * ((state.step / state.total_steps) ** 2)
        bound = (score - max_opp_score - buffer) // state.spawn_cost
        new_ships = max(bound, new_ships)

        # спавнится, если осталось много времени
        bound = len(YARD_SCHEDULE) * (state.step < SPAWNING_STEP)
        new_ships = max(bound, new_ships)

        # не появляются, если оно того не стоит и у нас есть несколько кораблей
        if (state.total_steps - state.step < STEPS_FINAL) and (ships >= 5):
            new_ships = 0

        # количество требуемых кораблей без ограничений
        self.ships_wanted = int(new_ships)

        # количество кораблей, которые мы можем построить с ограничениями
        possible = min(new_ships, state.my_halite // state.spawn_cost)
        self.ships_possible = min(int(possible), len(actions.yards))
        return

    def spawn(self, state, actions):
        # удалить spawn_pos, занятые кораблями после этого хода
        occupied = state.moved_this_turn[self.spawn_pos]
        free_pos = self.spawn_pos[~occupied]

        # получить идентификаторы верфей, которые должны появиться
        pos_to_yard = {v: k for k, v in state.my_yards.items()}
        ids = [pos_to_yard[pos] for pos in free_pos[0:self.ships_possible]]

        # записываем соответствующие действия в actions.decided
        actions.decided.update({yard: "SPAWN" for yard in ids})
        actions.yards.clear()
        return
