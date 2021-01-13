import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
from kaggle_environments import make


HALITE_GREEDY_ALGORITHM_BOT_FOLDER = "halite-greedy-algorithm-bot"
HALITE_IMITATION_LEARNING_BOT_FOLDER = "halite-imitation-learning-bot"
HALITE_SWARM_INTELLIGENCE_BOT_FOLDER = "halite-swarm-intelligence-bot"
HALITE_DEFENSIVE_GREEDY_BOT_FOLDER = "halite-defensive-greedy-bot"
HALITE_Q_LEARNING_BOT_FOLDER = "halite-q-learning-bot"

#    1    2
#    3    4
HALITE_BOT_ORDER_COLOR = [
    "#E2CD13",
    "#F24E4E",
    "#34BB1C",
    "#7B33E2"
]


def main():
    argv = sys.argv
    argc = len(argv)

    if (argc != 2 or not argv[1].isdigit()):
        print("\"steps\" parameter not passed or is a string.\nExample usage: 'python run.py 400'")
        exit(1)
    elif (int(argv[1]) > 400):
        print("Warning: you shouldn't attempt to run games with more than 400 steps")
        
    steps = int(argv[1])

    # вставить сюда путь к репозиторию из рабочего каталога
    path = os.curdir
    greedy_bot = os.path.join(
        path, HALITE_GREEDY_ALGORITHM_BOT_FOLDER, "main.py")
    imitation_learning_bot = os.path.join(
        path, HALITE_IMITATION_LEARNING_BOT_FOLDER, "main.py")
    swarm_intelligence_bot = os.path.join(
        path, HALITE_SWARM_INTELLIGENCE_BOT_FOLDER, "submission.py")
    defensive_greedy_bot = os.path.join(
        path, HALITE_DEFENSIVE_GREEDY_BOT_FOLDER, "main.py")
    q_learning_bot = os.path.join(path, HALITE_Q_LEARNING_BOT_FOLDER, "main.py")

    # может играть с 1, 2 или 4 игроками. None - неактивного агента, "random" -
    # встроенный случайный агент, или можно указать агентов по имени файла:
    # агент = [bot]
    # агенты = ["otheragent.py", bot]
    # агенты = ["random", bot, "random", None]
    agents = [
        {'bot': greedy_bot, 'name': "Жадный алгоритм (наш)"},
        {'bot': imitation_learning_bot, 'name': "Машинное обучение (имитационная модель)"},
        {'bot': defensive_greedy_bot, 'name': "\"Оборонительный\" жадный алгоритм"},
        {'bot': q_learning_bot, 'name': "Q-обучение с подкреплением"}
    ]

    agent_names = list(map(lambda a: a['name'], agents))
    agent_bots = list(map(lambda a: a['bot'], agents))

    # случайное начальное число для среды kaggle - установление значение None, чтобы пропустить
    # seed = None
    seed = None

    # добавить файлы в / src / к пути python
    greedy_src = os.path.join(path, HALITE_GREEDY_ALGORITHM_BOT_FOLDER)
    il_src = os.path.join(path, HALITE_IMITATION_LEARNING_BOT_FOLDER)
    si_src = os.path.join(path, HALITE_SWARM_INTELLIGENCE_BOT_FOLDER)
    q_src = os.path.join(path, HALITE_Q_LEARNING_BOT_FOLDER)

    sys.path.extend([greedy_src, il_src, si_src, q_src])

    # получить симулятор галита из окружения kaggle и запустить симуляцию
    if seed is not None:
        print(f"Running for {steps} steps with seed = {seed}...")
        config = {"randomSeed": seed, "episodeSteps": steps}
    else:
        print(f"Running for {steps} steps...")
        config = {"episodeSteps": steps}

    print("Setting up bots...")
    print()

    halite = make("halite", debug=True, configuration=config)
    played_match = halite.run(agent_bots)
    json.dump(played_match, fp=open("data.json", "w"),
              sort_keys=True, indent=4)

    make_graphs(played_match, agent_names)

    # рендеринг выходного видео в Simulation.html
    print("Rendering episode...")
    out = halite.render(mode="html", width=800, height=600)
    with open(os.path.join(path, "simulation.html"), "w") as file:
        file.write(out)

    # удалить файлы в / src / из пути Python
    sys.path.remove(greedy_src)
    sys.path.remove(il_src)
    sys.path.remove(si_src)
    sys.path.remove(q_src)

    print("Simulation done.\n")

    return


def make_graphs(played_match, bot_names):
    print(f"Generating graphs based on {len(played_match)} steps...\n")

    total_halite_during_match(played_match, bot_names)
    plt.clf()

    total_ships_during_match(played_match, bot_names)
    plt.clf()

    total_attack_ships_during_match(played_match, bot_names)
    plt.clf()

def prepare_plot_settings(bot_names):
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

    handles = []
    for i, bot_name in enumerate(bot_names):
        handles.append(mpatches.Patch(
            color=HALITE_BOT_ORDER_COLOR[i], label=bot_name))

    plt.legend(handles=handles, title="Агенты",
               loc='upper left', bbox_to_anchor=(0, 1.3))

    plt.xlabel('Шаг')


def total_halite_during_match(played_match, bot_names):
    prepare_plot_settings(bot_names)

    each_bot_halite_stats = [[] for _ in range(len(bot_names))]

    for step in played_match:
        step_data = step[0]
        players = step_data.observation.players

        for i in range(len(players)):
            each_bot_halite_stats[i].append(players[i][0])

    for i, halite_count in enumerate(each_bot_halite_stats):
        plt.plot(halite_count,
                 label=bot_names[i], color=HALITE_BOT_ORDER_COLOR[i])

    plt.ylabel('Количество ресурса (Halite)')

    plt.savefig("analysis/graph_total_halite_during_match.png", bbox_inches='tight')


def total_ships_during_match(played_match, bot_names):
    prepare_plot_settings(bot_names)

    each_bot_ship_stats = [[] for _ in range(len(bot_names))]

    for step in played_match:
        step_data = step[0]
        players = step_data.observation.players

        for i in range(len(players)):
            each_bot_ship_stats[i].append(len(players[i][2]))

    for i, ship_count in enumerate(each_bot_ship_stats):
        plt.plot(ship_count,
                 label=bot_names[i], color=HALITE_BOT_ORDER_COLOR[i])

    plt.ylabel('Количество кораблей')

    plt.savefig("analysis/graph_total_ships_during_match.png", bbox_inches='tight')

def total_attack_ships_during_match(played_match, bot_names):
    prepare_plot_settings(bot_names)

    each_bot_ship_stats = [[] for _ in range(len(bot_names))]

    def is_attack_ship(ship):
        return ship[1] < 10

    for step in played_match:
        step_data = step[0]
        players = step_data.observation.players

        for i in range(len(players)):
            all_ships = list(players[i][2].values())
            attack_ships = list(filter(is_attack_ship, all_ships))
            each_bot_ship_stats[i].append(len(attack_ships))

    for i, ship_count in enumerate(each_bot_ship_stats):
        plt.plot(ship_count,
                 label=bot_names[i], color=HALITE_BOT_ORDER_COLOR[i])

    plt.ylabel('Количество атакующих кораблей')

    plt.savefig("analysis/graph_total_attack_ships_during_match.png", bbox_inches='tight')


if __name__ == "__main__":
    main()
