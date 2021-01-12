import os
import sys
import json
import matplotlib.pyplot as plt

from kaggle_environments import make


HALITE_GREEDY_ALGORITHM_BOT_FOLDER = "halite-greedy-algorithm-bot"
HALITE_MACHINE_LEARNING_BOT_FOLDER = "halite-machine-learning-bot"
HALITE_SWARM_INTELLIGENCE_BOT_FOLDER = "halite-swarm-intelligence-bot"
HALITE_OTHER_GREEDY_BOT_FOLDER = "halite-other-greedy-bot"
HALITE_FARM_BOT_FOLDER = "halite-farm-bot"

#    1    2
#    3    4
HALITE_BOT_ORDER_COLOR = [
    "#E2CD13",
    "#F24E4E",
    "#34BB1C",
    "#7B33E2"
]


# количество шагов для запуска симуляции - стратегия бота
# имеет смысл только в том случае, если это 400, что было его значением во время
# конкурс на kaggle.com
STEPS = 400

def make_graphs(played_match, bot_names):
    each_bot_halite_stats = [[] for _ in range(len(bot_names))]

    for step_number, step in enumerate(played_match):
        step_data = step[0]
        players = step_data.observation.players
        
        for i in range(len(players)):
            each_bot_halite_stats[i].append(players[i][0])

    for i, halite_count in enumerate(each_bot_halite_stats):
        plt.plot(halite_count, label=bot_names[i], color=HALITE_BOT_ORDER_COLOR[i])
        
    legend = plt.legend(loc='upper right')
    # увеличить толщину линий
    for legobj in legend.legendHandles:
        legobj.set_linewidth(10.0)
        
    plt.ylabel('Количество ресурса (Halite)')
    plt.xlabel('Шаг')

    plt.savefig("graph.png")

def main():
    # вставить сюда путь к репозиторию из рабочего каталога
    path = os.curdir
    greedy_bot = os.path.join(path, HALITE_GREEDY_ALGORITHM_BOT_FOLDER, "main.py")
    machine_learning_bot = os.path.join(path, HALITE_MACHINE_LEARNING_BOT_FOLDER, "main.py")
    swarm_intelligence_bot = os.path.join(path, HALITE_SWARM_INTELLIGENCE_BOT_FOLDER, "submission.py")
    OTHER_GREEDY_bot = os.path.join(path, HALITE_OTHER_GREEDY_BOT_FOLDER, "main.py")
    farm_bot = os.path.join(path, HALITE_FARM_BOT_FOLDER, "farmBot.py")

    # может играть с 1, 2 или 4 игроками. None - неактивного агента, "random" -
    # встроенный случайный агент, или можно указать агентов по имени файла:
    # агент = [bot]
    # агенты = ["otheragent.py", bot]
    # агенты = ["random", bot, "random", None]
    agents = [
        { 'bot': greedy_bot, 'name': "Жадный алгоритм (наш) #1" }, 
        { 'bot': machine_learning_bot, 'name': "Машинное обучение" }, 
        { 'bot': OTHER_GREEDY_bot, 'name': "\"Оборонительный\" жадный алгоритм" },
        # { 'bot': swarm_intelligence_bot, 'name': "Роевой интеллект" }, 
        # { 'bot': greedy_bot, 'name': 'Жадный алгоритм (наш) #2'}
        { 'bot': farm_bot, 'name': "FarmBot"}
    ]

    agent_names = list(map(lambda a: a['name'], agents))
    agent_bots = list(map(lambda a: a['bot'], agents))

    # случайное начальное число для среды kaggle - установление значение None, чтобы пропустить
    # seed = None
    seed = None

    # добавить файлы в / src / к пути python
    greedy_src = os.path.join(path, HALITE_GREEDY_ALGORITHM_BOT_FOLDER)
    ml_src = os.path.join(path, HALITE_MACHINE_LEARNING_BOT_FOLDER)
    si_src = os.path.join(path, HALITE_SWARM_INTELLIGENCE_BOT_FOLDER)
    farm_bot = os.path.join(path, HALITE_FARM_BOT_FOLDER)

    sys.path.extend([greedy_src, ml_src, si_src, farm_bot])

    # получить симулятор галита из окружения kaggle и запустить симуляцию
    if seed is not None:
        print(f"Running for {STEPS} steps with seed = {seed}...")
        config = {"randomSeed": seed, "episodeSteps": STEPS}
    else:
        print(f"Running for {STEPS} steps...")
        config = {"episodeSteps": STEPS}

    print("Setting up bots...")
    print()

    halite = make("halite", debug=True, configuration=config)
    played_match = halite.run(agent_bots)

    make_graphs(played_match, agent_names)

    # рендеринг выходного видео в Simulation.html
    print("Rendering episode...")
    out = halite.render(mode="html", width=800, height=600)
    with open(os.path.join(path, "simulation.html"), "w") as file:
        file.write(out)

    # удалить файлы в / src / из пути Python
    sys.path.remove(greedy_src)
    sys.path.remove(ml_src)
    sys.path.remove(si_src)
    sys.path.remove(farm_bot)

    print("Done.")

    return


if __name__ == "__main__":
    main()
