import os
import sys
import json
import matplotlib.pyplot as plt

from kaggle_environments import make


HALITE_GREEDY_ALGORITHM_BOT_FOLDER = "halite-greedy-algorithm-bot"
HALITE_MACHINE_LEARNING_BOT_FOLDER = "halite-machine-learning-bot"


def main():
    # вставить сюда путь к репозиторию из рабочего каталога
    path = os.curdir
    greedy_bot = os.path.join(path, HALITE_GREEDY_ALGORITHM_BOT_FOLDER, "main.py")
    machine_learning_bot = os.path.join(path, HALITE_MACHINE_LEARNING_BOT_FOLDER, "main.py")

    # может играть с 1, 2 или 4 игроками. Нет - неактивного агента, "random" -
    # встроенный случайный агент, или можно указать агентов по имени файла:
    # агент = [bot]
    # агенты = ["otheragent.py", bot]
    # агенты = ["random", bot, "random", нет]
    agents = [greedy_bot, machine_learning_bot]

    # количество шагов для запуска симуляции - стратегия бота
    # имеет смысл только в том случае, если это 400, что было его значением во время
    # конкурс на kaggle.com
    steps = 400

    # случайное начальное число для среды kaggle - установление значение None, чтобы пропустить
    # seed = None
    seed = None

    # добавить файлы в / src / к пути python
    greedy_src = os.path.join(path, HALITE_GREEDY_ALGORITHM_BOT_FOLDER)
    ml_src = os.path.join(path, HALITE_MACHINE_LEARNING_BOT_FOLDER)

    sys.path.append(greedy_src)
    sys.path.append(ml_src)

    # получить симулятор галита из окружения kaggle и запустить симуляцию
    if seed is not None:
        print(f"Running for {steps} steps with seed = {seed}...")
        config = {"randomSeed": seed, "episodeSteps": steps}
    else:
        print(f"Running for {steps} steps...")
        config = {"episodeSteps": steps}

    halite = make("halite", debug=True, configuration=config)
    played_match = halite.run(agents)

    played_match_json_data = json.dumps(played_match, sort_keys=True, indent=2)

    print(played_match[1][0]["observation"]["players"][1][0])

    player0_halite, player1_halite = [], []

    for step_number, step in enumerate(played_match):
        step_data = step[0]
        players = step_data.observation.players
        
        for x, player in enumerate(players):
            print(f"[{step_number}] Player {x} has {player[0]} halite")

        player0_halite.append(players[0][0])
        player1_halite.append(players[1][0])

    with open(os.path.join(path, "data.json"), "w") as file:
        file.write(played_match_json_data)

    # рендеринг выходного видео в Simulation.html
    print("Rendering episode...")
    out = halite.render(mode="html", width=800, height=600)
    with open(os.path.join(path, "simulation.html"), "w") as file:
        file.write(out)

    # удалить файлы в / src / из пути Python
    sys.path.remove(greedy_src)
    sys.path.remove(ml_src)

    print("Done.")

    plt.plot(player0_halite, label="Greedy algorithm")
    plt.plot(player1_halite, label="Machine learning")
    plt.legend(loc='upper right')
    plt.ylabel('Halite')
    plt.xlabel('Step')
    plt.savefig("graph.png")

    return


if __name__ == "__main__":
    main()
