import numpy as np

from settings import (YARD_SCHEDULE, YARD_MAX_STEP, MIN_OPP_YARD_DIST,
                      MIN_YARD_DIST, SUPPORT_DIST, STEPS_INITIAL)


def convert(state, actions):
    # если у нас нет верфей, мы сразу пытаемся переделать корабль
    # с наибольшим количеством груза
    if len(state.my_yards) == 0:
        ship = max(actions.ships, key=lambda ship: state.my_ships[ship][1])

        if legal(ship, state):
            actions.decided[ship] = "CONVERT"
            state.update(ship, "CONVERT")
            actions.ships.remove(ship)

        return

    # в противном случае мы конвертируем корабль, если у нас слишком мало верфей для наших кораблей
    # и еще не поздно в игре(если у нас есть корабли)
    yards = working_yards(state)
    yards_wanted = sum([x <= state.my_ship_pos.size for x in YARD_SCHEDULE])
    should_convert = (yards.size < yards_wanted)
    should_convert = should_convert and (state.step < YARD_MAX_STEP)
    should_convert = should_convert and (len(actions.ships) > 0)

    if not should_convert:
        return

    # ограничить позиции, чтобы поблизости было как минимум 2 вспомогательные верфи
    supports = np.sum(state.dist[yards, :] <= SUPPORT_DIST, axis=0, initial=0)
    triangles = (supports >= min(yards.size, 2))

    # ограничить позиции, чтобы они имели минимальное расстояние до дружественных верфей
    closest = np.amin(state.dist[yards, :], axis=0, initial=state.map_size)
    triangles &= (closest >= MIN_YARD_DIST)

    # ограничить позиции, чтобы иметь минимальное расстояние до верфей соперника
    opp_yard_dist = np.amin(state.dist[state.opp_yard_pos, :], axis=0,
                            initial=state.map_size)
    triangles &= (opp_yard_dist >= MIN_OPP_YARD_DIST)

    # ограничить позиции, чтобы быть дальше, чем в одном шаге от кораблей противника
    opp_ship_dist = np.amin(state.dist[state.opp_ship_pos, :], axis=0,
                            initial=state.map_size)
    triangles &= (opp_ship_dist >= 2)

    
    # смотрим, какие корабли удовлетворяют этим ограничениям
    convertable = state.my_ship_pos[triangles[state.my_ship_pos]]

    if convertable.size == 0:
        return

    # найти корабль, вокруг которого больше всего галитовых ячеек
    halite = np.flatnonzero(state.halite_map > 0)
    hood = state.dist[np.ix_(halite, convertable)] <= 6
    cells = np.sum(hood, axis=0, initial=0)
    ship_pos = convertable[cells.argmax()]

    # и преобразовать этот корабль, если это разрешено
    pos_to_ship = {state.my_ships[ship][0]: ship for ship in actions.ships}
    ship = pos_to_ship.get(ship_pos, None)

    if (ship is not None) and legal(ship, state):
        actions.decided[ship] = "CONVERT"
        state.update(ship, "CONVERT")
        actions.ships.remove(ship)

    return


# returns True if CONVERT is a legal action for ship - need to have enough
# halite and not be on another yard. if you only have one ship, you can
# only convert if you still have enough halite to spawn a ship afterwards
# возвращает True, если CONVERT является законным действием для корабля - должно быть достаточно
# галита и не должен быть на другой верфи. Если у нас есть только один корабль, мы можем
# конвертировать, только если у нас достаточно галита, чтобы впоследствии создать корабль
def legal(ship, state):
    pos, hal = state.my_ships[ship]
    minhal = state.convert_cost - hal
    if len(state.my_ships) == 1:
        minhal += state.spawn_cost
    return (state.my_halite >= minhal) and (pos not in state.my_yard_pos)



# возвращает пригодные для использования верфи. Верфи непригодны, если есть
# верфь соперника с дистанцией 2
def working_yards(state):
    if state.step > STEPS_INITIAL:
        inds = np.ix_(state.opp_yard_pos, state.my_yard_pos)
        dists = np.amin(state.dist[inds], axis=0, initial=state.map_size)
        return state.my_yard_pos[dists > 2]
    else:
        return state.my_yard_pos
