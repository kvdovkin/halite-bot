import numpy as np
from scipy.optimize import linear_sum_assignment as assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from convert import working_yards
from settings import (BASELINE_SHIP_RATE, BASELINE_YARD_RATE, STEPS_INITIAL,
                      STEPS_SPIKE, RISK_PREMIUM, SPIKE_PREMIUM, MY_RADIUS,
                      MY_WEIGHT, OPP_RADIUS, OPP_WEIGHT)


class Targets:
    def __init__(self, state, actions, bounties, spawns):
        self.num_ships = len(actions.ships)

        # если кораблей нет, делать нечего
        if self.num_ships == 0:
            return

        # защищаем те верфи, которые работают, и в этом ходу у них не будет спауна
        likely_spawns = spawns.spawn_pos[0:spawns.ships_possible]
        yards = np.setdiff1d(working_yards(state), likely_spawns)

        # расстояние от ближайшего корабля противника до каждой верфи
        inds = np.ix_(state.opp_ship_pos, yards)
        opp_ship_dist = np.amin(state.dist[inds], axis=0,
                                initial=state.map_size)

        # расстояние ближайшего дружественного корабля к каждой верфи
        inds = np.ix_(state.my_ship_pos, yards)
        my_ship_dist = np.amin(state.dist[inds], axis=0,
                               initial=state.map_size)

        # если корабли противника начинают приближаться к верфи по сравнению
        # к своим, возвращаемся, чтобы защитить их
        inds = opp_ship_dist <= (2 + my_ship_dist)
        self.protected = yards[inds]
        self.protection_radius = opp_ship_dist[inds]

        # настраиваем возможные ходы для каждого корабля и вычисляем
        # расстояния на правильно взвешенном графике
        self.geometry(state, actions)

        # оптимальное назначение присвоит каждому месту только один корабль
        # но мы хотим, чтобы на каждую верфь вернулось более одного корабля, поэтому
        # мы добавляем дубликаты верфей к наградам, чтобы это стало возможным
        duplicates = np.tile(state.my_yard_pos, self.num_ships - 1)
        ind_to_site = np.append(duplicates, state.sites)

        # рассчитываем стоимость посещения места для каждого корабля
        cost_matrix = np.vstack([self.rewards(ship, state, bounties)
                                 for ship in actions.ships])

        # найти оптимальное назначение кораблей по направлениям
        # оптимальное назначение присваивает ship_inds [i] site_inds [i]
        ship_inds, site_inds = assignment(cost_matrix, maximize=True)

        # проходим решение задачи оптимального назначения и
        # упорядочиваем ходы по предпочтениям
        self.destinations = {}
        self.values = {}

        for ship_ind, site_ind in zip(ship_inds, site_inds):
            # сохранить пункт назначения и стоимость корабля
            ship = actions.ships[ship_ind]
            self.destinations[ship] = ind_to_site[site_ind]
            self.values[ship] = cost_matrix[ship_ind, site_ind]

            # sort перемещается по тому, насколько он уменьшает расстояние
            # в назначенный пункт назначения
            dest_dists = self.move_dists[ship][:, self.destinations[ship]]
            self.moves[ship] = self.moves[ship][dest_dists.argsort()]

        return

    def rates(self, state, ship):
        pos, hal = state.my_ships[ship]

        SR = BASELINE_SHIP_RATE
        YR = BASELINE_YARD_RATE

        threats = (state.opp_ship_hal < hal)

        # вначале учитывайте угрозы для близких кораблей
        if state.step < STEPS_INITIAL:
            threats &= (state.dist[state.opp_ship_pos, pos] <= 4)

        YR += RISK_PREMIUM * np.sum(threats)
        SR += RISK_PREMIUM * np.sum(threats)

        # сделать ставку огромной в конце игры (корабли должны вернуться домой)
        if state.total_steps - state.step < STEPS_SPIKE:
            YR += SPIKE_PREMIUM
            SR += SPIKE_PREMIUM

        # убедитесь, что все ставки <1, чтобы формулы оставались стабильными
        SR = min(SR, 0.9)
        YR = min(YR, 0.9)
        return SR, YR

    def rewards(self, ship, state, bounties):
        pos, hal = state.my_ships[ship]

        reward_map = np.zeros_like(state.halite_map)

        pos_ind = np.flatnonzero(self.moves[ship] == pos)[0]
        ship_dists = self.move_dists[ship]
        ship_dists = ship_dists[pos_ind, :]

        # определить скорость корабля, скорость верфи и скорость охоты
        SR, YR = self.rates(state, ship)

        # индексы добываемых участков - игнорировать галит рядом с вражескими верфями
        opp_yard_dist = np.amin(state.dist[state.opp_yard_pos, :], axis=0,
                                initial=state.map_size)
        minable = (state.halite_map > 0) & (opp_yard_dist > 1)

        # добавить награды за майнинг на всех майнинговых местах
        H = state.halite_map[minable]
        SD = ship_dists[minable]
        YD = self.yard_dists[ship][minable]

        beta = (1 + state.regen_rate) * (1 - state.collect_rate)
        alpha = state.collect_rate / (1 - beta)

        F = ((1 + SR) ** SD) * ((1 + YR) ** YD)
        F1 = hal / F
        F2 = alpha * ((1 + state.regen_rate) ** SD) * H
        F2 = F2 / F

        # вычисляем максимизирующую M
        M = np.log(1 + F1 / F2) - np.log(1 - np.log(beta) / np.log(1 + YR))
        M = np.fmax(1, np.round(M / np.log(beta)))

        # вставить награды за майнинг
        reward_map[minable] = (F1 + F2 * (1 - beta ** M)) / ((1 + YR) ** M)

        # добавить награды на верфи за хранение галита
        ship_yard_dist = ship_dists[state.my_yard_pos]
        reward_map[state.my_yard_pos] = hal / ((1 + YR) ** ship_yard_dist)

        # вставить награды за корабли противника
        ship_hunt_pos, ship_hunt_rew = bounties.get_ship_targets(ship, state)
        discount = (1 + SR) ** ship_dists[ship_hunt_pos]
        discount = discount * (1 + YR) ** self.yard_dists[ship][ship_hunt_pos]
        reward_map[ship_hunt_pos] = ship_hunt_rew / discount

        # скопируйте награды верфи на дублирующие верфи и
        # добавить повторяющиеся награды
        yard_rewards = reward_map[state.my_yard_pos]
        duplicate_rewards = np.tile(yard_rewards, self.num_ships - 1)

        # наконец добавить большой бонус к любым защищенным верфям, которые
        # корабль может защитить - это гарантирует, что один близкий корабль всегда
        # выбирает верфь - но не копирует этот бонус на дубликаты
        # так как там должен быть только один корабль
        inds = state.dist[self.protected, pos] < self.protection_radius
        reward_map[self.protected[inds]] += 1000

        return np.append(duplicate_rewards, reward_map)

    def geometry(self, state, actions):
        self.moves = {}
        self.move_dists = {}
        self.yard_dists = {}

        for ship in actions.ships:
            pos, hal = state.my_ships[ship]
            self.moves[ship] = np.flatnonzero(state.dist[pos, :] <= 1)

            # построить взвешенный график для расчета расстояний
            graph = self.make_graph(ship, state)

            # расстояние до каждого места после перемещения
            self.move_dists[ship] = dijkstra(graph, indices=self.moves[ship])

            # рассчитываем расстояния от всех участков до ближайшей верфи
            # если нет верфей, возьмите вместо этого максимальный грузовой корабль
            # какой корабль, скорее всего, скоро конвертируется
            if state.my_yard_pos.size != 0:
                yard_pos = state.my_yard_pos
            else:
                yard_pos = state.my_ship_pos[state.my_ship_hal.argmax()]

            # расстояние до ближайшей дружественной верфи
            self.yard_dists[ship] = dijkstra(graph, indices=yard_pos,
                                             min_only=True)

        return

    def make_graph(self, actor, state):
        pos, hal = state.my_ships[actor]
        weights = np.ones_like(state.sites)

        # игнорировать непосредственных соседей в дружественных весах - это обрабатывается
        # автоматически и вес может вызвать пробки с близкого расстояния
        # также игнорировать вес любых кораблей на верфях
        friendly = np.setdiff1d(state.my_ship_pos, state.my_yard_pos)
        friendly = friendly[state.dist[pos, friendly] > 1]
        hood = state.dist[friendly, :] <= MY_RADIUS
        weights += MY_WEIGHT * np.sum(hood, axis=0, initial=0)

        # рассматривать только корабли противника с меньшим количеством галита
        threats = state.opp_ship_pos[state.opp_ship_hal < hal]
        hood = state.dist[threats, :] <= OPP_RADIUS
        weights += OPP_WEIGHT * np.sum(hood, axis=0, initial=0)

        # также нужно обойти верфи противника
        weights[state.opp_yard_pos] += OPP_WEIGHT

        # убрать грузы с верфей, чтобы они не блокировались
        weights[state.my_yard_pos] = 0

        # индексы столбца для строки i находятся в индексах [indptr [i]: indptr [i + 1]]
        # и соответствующие им значения находятся в data [indptr [i]: indptr [i + 1]]
        nsites = state.map_size ** 2
        indptr = 4 * np.append(state.sites, nsites)

        indices = np.empty(4 * nsites, dtype=int)
        indices[0::4] = state.north
        indices[1::4] = state.south
        indices[2::4] = state.east
        indices[3::4] = state.west

        # вес на любом ребре (x, y) равен (w [x] + w [y]) / 2
        data = np.empty(4 * nsites, dtype=float)
        data[0::4] = 0.5 * (weights + weights[state.north])
        data[1::4] = 0.5 * (weights + weights[state.south])
        data[2::4] = 0.5 * (weights + weights[state.east])
        data[3::4] = 0.5 * (weights + weights[state.west])

        return csr_matrix((data, indices, indptr), shape=(nsites, nsites))
