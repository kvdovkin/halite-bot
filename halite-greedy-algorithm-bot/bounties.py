import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from settings import (SHIPS_PER_BOUNTY, HUNTING_MAX_RATIO, HUNTING_STEP,
                      HUNT_WEIGHT, HUNT_RADIUS, HUNT_NEARBY, STEPS_SPIKE)


class Bounties:
    def __init__(self, state, ship_target_memory):
        self.num_targets = int(state.my_ship_pos.size // SHIPS_PER_BOUNTY)

        # вычисление позиции, галита и уязвимости для всех кораблей противника
        # уязвимость - это отношение расстояния до ближайшей дружественной верфи
        # на взвешенном графике по расстоянию до ближайшего дружественной верфи на
        # графике с постоянными весами, равными mean_weight. уязвимость
        # больше единицы означает, что у нас есть охотники, преграждающие путь к
        # ближайшей верфи ...
        opp_ship_pos = np.array([], dtype=int)
        opp_ship_hal = np.array([], dtype=int)
        opp_ship_vul = np.array([], dtype=float)
        opp_ship_dis = np.array([], dtype=int)

        graph = self.make_graph(state)

        for opp in state.opp_data.values():
            yards, ship_pos, ship_hal = opp[1:4]

            if yards.size == 0:
                ship_vul = np.full_like(ship_pos, 10)
                ship_dis = np.full_like(ship_pos, 10)
            else:
                graph_dist = dijkstra(graph, indices=yards, min_only=True)
                graph_dist = graph_dist[ship_pos]
                ship_dis = np.amin(state.dist[np.ix_(yards, ship_pos)], axis=0)
                ship_vul = (1 + graph_dist) / (1 + ship_dis)

            opp_ship_pos = np.append(opp_ship_pos, ship_pos)
            opp_ship_hal = np.append(opp_ship_hal, ship_hal)
            opp_ship_vul = np.append(opp_ship_vul, ship_vul)
            opp_ship_dis = np.append(opp_ship_dis, ship_dis)

        # nearby содержит количество охотников на расстоянии 3
        # которые также имеют строго меньше груза, чем корабль
        nearby = state.dist[np.ix_(state.my_ship_pos, opp_ship_pos)] <= 3
        less_hal = state.my_ship_hal[:, np.newaxis] < opp_ship_hal
        nearby = np.sum(nearby & less_hal, axis=0)

        # сохранение текущий позиции предыдущих целей, которые еще живы
        prev = np.array([val[0] for key, val in state.opp_ships.items()
                         if key in ship_target_memory], dtype=int)

        # получение индексов уже нацеленных кораблей
        # если корабль находится слишком близко к дружественной верфи, он, вероятно, ускользнет
        # так что убираем такие корабли с целей
        target_bool = np.in1d(opp_ship_pos, prev) & (opp_ship_dis > 2)
        target_inds = np.flatnonzero(target_bool)

        
        # список возможных новых целей состоит из нецелевых кораблей
        # в ловушке (уязвимость> 1) иметь хотя бы одного охотника
        # рядом и не слишком близко к дружественной верфи
        candidates = ~target_bool & (opp_ship_vul > 1)
        candidates &= (opp_ship_dis > 2) & (nearby >= HUNT_NEARBY)

        # мы подсчитываем баллы для каждого корабля-кандидата, указывая
        # риск / награда за нападение на них
        # сделать оценки кораблей, которые не являются кандидатами, отрицательны
        opp_ship_score = opp_ship_hal * opp_ship_vul
        opp_ship_score[~candidates] = -1

        # определение, сколько целей мы хотели бы иметь и сколько
        # новых целей, которые мы должны / можем построить. мы ставим новые цели только если
        # осталось не так много галита, который мы могли бы добыть - однако
        # мы всегда охотимся после первой части игры
        ratio = np.sum(state.halite_map) / state.starting_halite

        if (ratio > HUNTING_MAX_RATIO) and (state.step < HUNTING_STEP):
            num_new_targets = 0
        else:
            num_new_targets = max(self.num_targets - target_inds.size, 0)
            num_new_targets = min(num_new_targets, np.sum(candidates))

        # мы можем взять (num_new_targets) корабли с максимальным количеством очков
        # так как баллы >= 0 и мы принудительно выставили баллы некандидату,
        # корабль равен -1
        new_inds = np.argpartition(-opp_ship_score, num_new_targets - 1)
        target_inds = np.append(target_inds, new_inds[0:num_new_targets])

        # устанавливаем позицию / галит / награды для целей
        self.ship_targets_pos = opp_ship_pos[target_inds]
        self.ship_targets_hal = opp_ship_hal[target_inds]
        self.ship_targets_rew = np.full_like(self.ship_targets_pos,
                                             state.max_halite)

        # записываем новые цели в список ship_targets
        self.target_list = [key for key, val in state.opp_ships.items()
                            if val[0] in self.ship_targets_pos]

        return

    def get_ship_targets(self, ship, state):
        full_pos = np.array([], dtype=int)
        full_rew = np.array([], dtype=int)

        
      # прекратить охоту на корабли после скачка процентных ставок
        if state.total_steps - state.step < STEPS_SPIKE:
            return full_pos, full_rew

        # нахождение целей, которые мы можем атаковать
        ship_hal = state.my_ships[ship][1]
        attackable = self.ship_targets_hal > ship_hal
        targets_pos = self.ship_targets_pos[attackable]
        targets_hal = self.ship_targets_hal[attackable]
        targets_rew = self.ship_targets_rew[attackable]

        # расставление наград вокруг кораблей, на которых установлена цель
        for pos, hal, rew in zip(targets_pos, targets_hal, targets_rew):
            hood = np.flatnonzero(state.dist[pos, :] <= 1)
            hood_rew = np.full_like(hood, rew)
            full_pos = np.append(full_pos, hood)
            full_rew = np.append(full_rew, hood_rew)

        # удаление всех повторяющихся индексов и наград
        full_pos, inds = np.unique(full_pos, return_index=True)
        full_rew = full_rew[inds]
        return full_pos, full_rew

    def make_graph(self, state):
        nsites = state.map_size ** 2
        weights = np.ones(nsites)

        # узнаем, какие корабли могут охотиться на других и
        # добавляем их веса на график
        num_hunters = 5 * self.num_targets
        likely_hunters = np.argpartition(state.my_ship_hal, num_hunters - 1)
        likely_hunters = likely_hunters[0:num_hunters]
        hunters_pos = state.my_ship_pos[likely_hunters]
        hood = state.dist[hunters_pos, :] <= HUNT_RADIUS
        weights += HUNT_WEIGHT * np.sum(hood, axis=0, initial=0)

        # вес на любом ребре (x, y) равен (w [x] + w [y]) / 2
        # индексы столбца для строки i находятся в индексах [indptr [i]: indptr [i + 1]]
        # и соответствующие им значения находятся в data [indptr [i]: indptr [i + 1]]
        indptr = 4 * np.append(state.sites, nsites)

        indices = np.empty(4 * nsites, dtype=int)
        indices[0::4] = state.north
        indices[1::4] = state.south
        indices[2::4] = state.east
        indices[3::4] = state.west

        data = np.empty(4 * nsites, dtype=float)
        data[0::4] = 0.5 * (weights + weights[state.north])
        data[1::4] = 0.5 * (weights + weights[state.south])
        data[2::4] = 0.5 * (weights + weights[state.east])
        data[3::4] = 0.5 * (weights + weights[state.west])

        return csr_matrix((data, indices, indptr), shape=(nsites, nsites))
