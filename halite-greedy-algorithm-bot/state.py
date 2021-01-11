import numpy as np


class State:
    def __init__(self, obs, config):
        # прочитать конфигурацию игры
        self.map_size = config.size
        self.total_steps = config.episodeSteps
        self.starting_halite = config.startingHalite
        self.regen_rate = config.regenRate
        self.collect_rate = config.collectRate
        self.convert_cost = config.convertCost
        self.spawn_cost = config.spawnCost
        self.max_halite = config.maxCellHalite

        # шаг и карта галита
        self.step = obs.step
        self.halite_map = np.array(obs.halite)

        # наш галит, верфи и корабли
        self.my_id = obs.player
        self.my_halite, self.my_yards, self.my_ships = obs.players[self.my_id]

        # установить совместные и индивидуальные данные для оппонентов
        self.set_opp_data(obs)

        # нескольким функциям нужен вектор всех мест, поэтому мы создаем только
        # один раз и сохраняем
        self.sites = np.arange(self.map_size ** 2)

        # список позиций с кораблями, которые уже двинулись в этот ход
        self.moved_this_turn = np.full(self.map_size ** 2, False, dtype=bool)

        # справочная таблица для эффекта ходов
        # север [x] - позиция к северу от x и т. д.
        self.north = (self.sites - self.map_size) % (self.map_size ** 2)
        self.south = (self.sites + self.map_size) % (self.map_size ** 2)
        self.east = self.sites + 1
        self.east[(self.map_size - 1)::self.map_size] -= self.map_size
        self.west = self.sites - 1
        self.west[0::self.map_size] += self.map_size

        # dist [x, y] хранит расстояние l1 между x и y на торе.
        cols = self.sites % self.map_size
        rows = self.sites // self.map_size
        coldist = cols - cols[:, np.newaxis]
        rowdist = rows - rows[:, np.newaxis]
        coldist = np.fmin(np.abs(coldist), self.map_size - np.abs(coldist))
        rowdist = np.fmin(np.abs(rowdist), self.map_size - np.abs(rowdist))
        self.dist = coldist + rowdist

        # устанавливает количество массивов numpy, производных от self.my_ships и т.д.
        self.set_derived()
        return

    def set_opp_data(self, obs):
        # список идентификаторов оппонентов
        self.opp_ids = list(range(0, len(obs.players)))
        self.opp_ids.remove(self.my_id)

        # совместные корабли и верфи противника
        self.opp_ships = {}
        self.opp_yards = {}
        for opp in self.opp_ids:
            self.opp_yards.update(obs.players[opp][1])
            self.opp_ships.update(obs.players[opp][2])

        # массивы, содержащие общие данные о судне / верфи для всех противников
        poshal = np.array(list(self.opp_ships.values()), dtype=int)
        pos, hal = np.hsplit(poshal, 2)
        self.opp_ship_pos = np.ravel(pos)
        self.opp_ship_hal = np.ravel(hal)
        self.opp_yard_pos = np.array(list(self.opp_yards.values()), dtype=int)

        # построим список списков с галитом, верфями, кораблем
        # позиции, отправляем галит для каждого противника в виде множества массивов
        self.opp_data = {}
        self.opp_scores = {}
        self.opp_num_ships = {}

        for opp in self.opp_ids:
            halite, yards, ships = obs.players[opp]

            poshal = np.array(list(ships.values()), dtype=int)
            ship_pos, ship_hal = np.hsplit(poshal, 2)
            ship_pos = np.ravel(ship_pos)
            ship_hal = np.ravel(ship_hal)
            yard_pos = np.array(list(yards.values()), dtype=int)

            self.opp_data[opp] = [halite, yard_pos, ship_pos, ship_hal]
            self.opp_num_ships[opp] = ship_pos.size
            if ship_pos.size + yard_pos.size > 0:
                self.opp_scores[opp] = halite + np.sum(ship_hal)
            else:
                self.opp_scores[opp] = 0

        return

    # several function need all our ship/yard positions as numpy arrays
    # these arrays need to be set by init() and also updated by update()
    # do this by calling set_derived()
    # нескольким функциям нужны все наши позиции корабля / верфи в виде массивов numpy
    # эти массивы должны быть установлены с помощью init (), а также обновлены с помощью update ()
    # делаем это, вызывая set_dehibited ()
    def set_derived(self):
        poshal = np.array(list(self.my_ships.values()), dtype=int)
        pos, hal = np.hsplit(poshal, 2)
        self.my_ship_pos = np.ravel(pos)
        self.my_ship_hal = np.ravel(hal)
        self.my_yard_pos = np.array(list(self.my_yards.values()), dtype=int)
        return

    def pos_to_move(self, initial, final):
        if final == self.north[initial]:
            return "NORTH"
        elif final == self.south[initial]:
            return "SOUTH"
        elif final == self.east[initial]:
            return "EAST"
        elif final == self.west[initial]:
            return "WEST"
        else:
            return None

    def move_to_pos(self, initial, move):
        if move == "NORTH":
            return self.north[initial]
        elif move == "SOUTH":
            return self.south[initial]
        elif move == "EAST":
            return self.east[initial]
        elif move == "WEST":
            return self.west[initial]
        else:
            return initial

    def update(self, actor, action):
        # если actor верфь, то только спаун влияет на состояние
        if (actor in self.my_yards) and (action == "SPAWN"):
            # создаем новую строку идентификатора
            newid = f"spawn[{actor}]"

            # создать новый корабль без груза на верфи
            pos = self.my_yards[actor]
            self.my_ships[newid] = [pos, 0]

            # результат здесь - корабль, который не может двигаться в этот ход
            self.moved_this_turn[pos] = True

            # вычесть стоимость появления из доступного галита
            self.my_halite -= int(self.spawn_cost)

        if actor in self.my_ships:
            pos, hal = self.my_ships[actor]

            if action == "CONVERT":
                # создать новую верфь на позиции корабля и удалить корабль
                self.my_yards[actor] = pos
                del self.my_ships[actor]

                self.my_halite += min(hal - self.convert_cost, 0)
                self.halite_map[pos] = 0

            else:
                # если корабль стоит на месте, он может собирать галит
                if action is None:
                    collect = self.halite_map[pos] * self.collect_rate
                    nhal = int(hal + collect)
                    self.halite_map[pos] -= collect
                else:
                    nhal = hal

                # записываем новую позицию и галит в my_ships
                npos = self.move_to_pos(pos, action)
                self.my_ships[actor] = [npos, nhal]

                # добавить новую позицию для кораблей, которые больше не могут двигаться в этот ход
                self.moved_this_turn[npos] = True

        # обновить внутренние переменные, производные от my_ships, my_yards
        self.set_derived()
        return
