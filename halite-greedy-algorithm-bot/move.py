import numpy as np
from scipy.optimize import linear_sum_assignment as assignment

from convert import working_yards
from settings import STEPS_SPIKE


def move(state, actions, targets):
    # если нет ожидающих кораблей, то делать нечего
    if len(actions.ships) == 0:
        return

    # рассчитываем стоимость за каждый шаг, идя к определенному месту
    cost_matrix, threat_matrix, threat_scores = matrices(state, actions,
                                                         targets)

    # упорядочим бесконечности - заменим их очень отрицательными конечными значениями
    # чтобы никогда не могло быть компенсировано хорошим соответствием. так что любое соответствие
    # с эффективной бесконечностью хуже, чем любое сопоставление без нее
    finite = np.isfinite(cost_matrix)
    infinite = ~finite
    eff_inf = 1 + 2 * len(actions.ships) * np.max(np.abs(cost_matrix[finite]))
    cost_matrix[infinite] = -eff_inf

    # находим оптимальное присваивание кораблей к местам
    ship_inds, sites = assignment(cost_matrix, maximize=True)

    # найти решение - если назначенное место законно и безопасно
    # переходим на него, иначе добавляем корабль в список кораблей
    # по которым решения принимаются независимо
    threatened = []
    depositing = []

    for ship_ind, site in zip(ship_inds, sites):
        ship = actions.ships[ship_ind]
        pos, hal = state.my_ships[ship]

        
        # если корабль был назначен на небезопасный объект, решаем переехать позже
        # если место не является охраняемой верфью
        if infinite[ship_ind, site] or threat_matrix[ship_ind, site]:
            threatened.append(ship_ind)
            continue
      
        # если корабль делает депозит после всплеска процентов и
        # в верфи движение, передвигаемся свободно
        spike = (state.total_steps - state.step) < STEPS_SPIKE
        spike = spike and (state.my_yard_pos.size > 0)
        if spike:
            yard_ind = np.argmin(state.dist[state.my_yard_pos, pos], axis=0)
            yard = state.my_yard_pos[yard_ind]
            close = state.dist[yard, pos] <= 3
            traffic = np.sum(state.dist[state.my_ship_pos, yard] <= 2) >= 6
            if traffic and close:
                depositing.append(ship_ind)
                continue

        decision = state.pos_to_move(pos, site)
        actions.decided[ship] = decision
        state.update(ship, decision)

    # принимаем решение о действиях для кораблей, которые были отнесены к небезопасным местам
    for ship_ind in threatened:
        ship = actions.ships[ship_ind]
        pos, hal = state.my_ships[ship]

        # ограничиваться местами с наименьшим количеством угроз
        legal = np.flatnonzero(state.dist[pos, :] <= 1)
        scores = threat_scores[ship_ind, legal]
        candidates = legal[scores == scores.min()]

        # далее ограничиваем места с наименьшим количеством столкновений с противниками
        scores = state.moved_this_turn[candidates]
        candidates = candidates[scores == scores.min()]

        # из них выбираем места с наивысшим рейтингом
        ranking = targets.moves[ship]
        ind = np.in1d(ranking, candidates).argmax()
        site = ranking[ind]
        decision = state.pos_to_move(pos, site)

        # если у нас нет безопасных площадок, куда можно было бы пойти, и еще груза
        # чем затраты на преобразование, преобразование и сохранение разницы
        if threat_matrix[ship_ind, site] and (hal >= state.convert_cost):
            decision = "CONVERT"

        actions.decided[ship] = decision
        state.update(ship, decision)

    for ship_ind in depositing:
        ship = actions.ships[ship_ind]
        pos, hal = state.my_ships[ship]

        # проверяем только на угрозы, а не на столкновения
        legal = np.flatnonzero(state.dist[pos, :] <= 1)
        scores = threat_scores[ship_ind, legal]
        candidates = legal[scores == scores.min()]

        ranking = targets.moves[ship]
        ind = np.in1d(ranking, candidates).argmax()
        site = ranking[ind]

        decision = state.pos_to_move(pos, site)
        actions.decided[ship] = decision
        state.update(ship, decision)

    actions.ships.clear()
    return


def matrices(state, actions, targets):
    dims = (len(actions.ships), state.map_size ** 2)
    threat_matrix = np.full(dims, False, dtype=bool)
    threat_scores = np.zeros(dims)
    cost_matrix = np.zeros(dims)

    # сохраняем истинное (l1) расстояние до пунктов назначения для каждого корабля
    dists_to_dest = np.zeros(len(actions.ships))
    for index in range(len(actions.ships)):
        ship = actions.ships[index]
        pos, hal = state.my_ships[ship]
        dest = targets.destinations[ship]
        dists_to_dest[index] = state.dist[pos, dest]

    # построить cost_matrix и threat_matrix
    for index in range(len(actions.ships)):
        ship = actions.ships[index]
        pos, hal = state.my_ships[ship]
        dest = targets.destinations[ship]
        
        # найти те корабли, у которых меньше галита, чем у нас
        # добавляем 1 к hal, если мы хотим иметь строгое сравнение галита
        # поскольку x <hal становится x <= hal для целочисленных значений ...
        strict = (pos not in state.my_yard_pos)
        ships = state.opp_ship_pos[state.opp_ship_hal < (hal + strict)]
        ship_dist = np.amin(state.dist[ships, :], axis=0,
                            initial=state.map_size)

        # угрожаемые места - это верфь противника и площадки, где корабли
        # За один шаг могут перевезти меньше груза
        threat_matrix[index, state.opp_yard_pos] = True
        threat_matrix[index, (ship_dist <= 1)] = True
        threat_matrix[index, working_yards(state)] = False

        weak_ships = state.opp_ship_pos[state.opp_ship_hal < hal]
        weak_ship_hood = state.dist[weak_ships, :] <= 1
        threat_scores[index, :] += np.sum(weak_ship_hood, axis=0, initial=0)
        threat_scores[index, state.opp_yard_pos] = 2

        # наказываем правильные ходы, ранжируя цели
        cost_matrix[index, targets.moves[ship]] = -10 * np.arange(5)

        # стараемся не собирать ненужный груз, который делает корабли уязвимыми
        # поэтому добавляем дополнительный штраф к текущей позиции
        no_cargo = (dest not in state.my_yard_pos)
        no_cargo = no_cargo and (state.halite_map[pos] > 0)
        no_cargo = no_cargo and (pos != dest)
        cost_matrix[index, pos] -= (100 if no_cargo else 0)
       
        # штрафуем за посещение небезопасных площадей
        cost_matrix[index, threat_matrix[index, :]] -= 1000
      
        # отдавать более высокий приоритет судам с большим грузом, но с наибольшим
        # приоритет для судов без груза
        if hal == 0:
            multiplier = 3
        else:
            rank = np.sum(hal > state.my_ship_hal)
            multiplier = 1 + rank / state.my_ship_hal.size

        # разорвать связи по расстоянию до пункта назначения
        dist = dists_to_dest[index]
        rank = np.sum(dist < dists_to_dest)
        multiplier += (rank / dists_to_dest.size) / 10

        cost_matrix[index, :] = multiplier * cost_matrix[index, :]

        # наказывать неправильные ходы бесконечностью
        cost_matrix[index, (state.dist[pos, :] > 1)] = -np.inf

    return cost_matrix, threat_matrix, threat_scores
