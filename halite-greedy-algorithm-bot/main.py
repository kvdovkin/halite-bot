# импортируем исходные файлы
from bounties import Bounties
from convert import convert
from move import move
from spawns import Spawns
from state import State
from targets import Targets


# объект для хранения ожидающих решений кораблей / верфей, а также
# действие, которое мы определили
class Actions:
    def __init__(self, state):
        self.decided = {}
        self.ships = list(state.my_ships)
        self.yards = list(state.my_yards)
        return

    def asdict(self):
        return {k: v for k, v in self.decided.items() if v is not None}


# глобальная переменная, в которой хранятся корабли противника, за которыми мы охотимся между ходами
ship_target_memory = []


def agent(obs, config):
    # читаем (obs, config) во внутренний объект состояния игры
    state = State(obs, config)

    # Объект actions хранит список ожидающих кораблей / верфей. 
    # после решения, мы удаляем корабли / верфи из списков ожидания и сохраняем
    # их в словаре вместе с их действиями
    actions = Actions(state)

    
    # преобразовать подходящие корабли в верфи
    convert(state, actions)

    # планируем, где мы хотим создавать новые корабли
    spawns = Spawns(state, actions)

    # размещение награды за выбранные корабли / верфи противника и запоминание
    # за какие корабли мы назначили награду на будущее
    global ship_target_memory
    bounties = Bounties(state, ship_target_memory)
    ship_target_memory = bounties.target_list

    # задаем пункты назначения для кораблей и ранжируем их, насколько ближе
    # мы добираемся до пунктов назначения
    targets = Targets(state, actions, bounties, spawns)

    # решаем ходы кораблей
    move(state, actions, targets)

    # создаем новые корабли на незанятых верфях
    spawns.spawn(state, actions)

    return actions.asdict()
