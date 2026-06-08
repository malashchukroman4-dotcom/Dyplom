import multiprocessing as mp
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


# =========================================================
# СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ
# =========================================================
# Приклад:
#
# x^2 + y^2 - 25 = 0
# x - cos(y) = 0
#
# Шукаємо x, y, щоб обидва рівняння були близькі до 0
# =========================================================

def equations(x, y):
    eq1 = x**2 + y**2 - 25
    eq2 = x * y - 12

    return eq1, eq2


# =========================================================
# TOTAL ERROR
# =========================================================

def total_error(x, y):
    eq1, eq2 = equations(x, y)

    return np.sqrt(eq1**2 + eq2**2)


# =========================================================
# FITNESS FUNCTION
# =========================================================

def fitness(x, y):
    error = total_error(x, y)

    return 1 / (1 + error)


# =========================================================
# ПАРАМЕТРИ
# =========================================================

@dataclass
class GAConfig:
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    migration_interval: int
    migration_size: int
    islands: int

    x_min: float
    x_max: float

    y_min: float
    y_max: float

    topology: str
    target_error: float


# =========================================================
# ІНДИВІД
# =========================================================

@dataclass
class Individual:
    x: float
    y: float
    fit: float


# =========================================================
# СТВОРЕННЯ ПОПУЛЯЦІЇ
# =========================================================

def create_population(config):
    population = []

    for _ in range(config.population_size):
        x = random.uniform(config.x_min, config.x_max)
        y = random.uniform(config.y_min, config.y_max)

        population.append(
            Individual(
                x,
                y,
                fitness(x, y)
            )
        )

    return population


# =========================================================
# TOURNAMENT SELECTION
# =========================================================

def tournament_selection(population, k=3):
    selected = random.sample(population, k)

    return max(selected, key=lambda ind: ind.fit)


# =========================================================
# CROSSOVER
# =========================================================

def crossover(parent1, parent2, config):
    if random.random() < config.crossover_rate:
        alpha = random.random()

        child_x = alpha * parent1.x + (1 - alpha) * parent2.x
        child_y = alpha * parent1.y + (1 - alpha) * parent2.y

    else:
        child_x = parent1.x
        child_y = parent1.y

    return Individual(
        child_x,
        child_y,
        fitness(child_x, child_y)
    )


# =========================================================
# MUTATION
# =========================================================

def mutate(individual, config):
    if random.random() < config.mutation_rate:
        individual.x += random.gauss(0, 1)
        individual.y += random.gauss(0, 1)

        individual.x = max(
            config.x_min,
            min(config.x_max, individual.x)
        )

        individual.y = max(
            config.y_min,
            min(config.y_max, individual.y)
        )

        individual.fit = fitness(individual.x, individual.y)

    return individual


# =========================================================
# EVOLUTION OF ONE ISLAND
# =========================================================

def evolve_island(
        island_id,
        config,
        send_pipes,
        recv_pipes,
        result_queue
):

    population = create_population(config)

    best_history = []
    avg_history = []
    migration_history = []
    iteration_times = []
    error_history = []

    for generation in range(config.generations):

        iter_start = time.time()

        new_population = []

        while len(new_population) < config.population_size:

            p1 = tournament_selection(population)
            p2 = tournament_selection(population)

            child = crossover(p1, p2, config)
            child = mutate(child, config)

            new_population.append(child)

        population = sorted(
            new_population,
            key=lambda ind: ind.fit,
            reverse=True
        )

        best = population[0]

        best_history.append(best.fit)

        avg_history.append(
            np.mean([ind.fit for ind in population])
        )

        error = total_error(best.x, best.y)
        error_history.append(error)

        iter_end = time.time()
        iteration_time = iter_end - iter_start
        iteration_times.append(iteration_time)

        eq1, eq2 = equations(best.x, best.y)

        print(
            f"Island {island_id} | "
            f"Generation {generation} | "
            f"Best fitness = {best.fit:.10f} | "
            f"Error = {error:.10e} | "
            f"Time = {iteration_time:.6f} sec"
        )

        # =================================================
        # КРИТЕРІЙ ЗУПИНКИ ПРИ ДОСЯГНЕННІ ТОЧНОСТІ
        # =================================================

        if error < config.target_error:
            print("\n======================================")
            print(f"Island {island_id}: TARGET ACCURACY REACHED")
            print("======================================")
            print(f"Generation = {generation}")
            print(f"Best x = {best.x:.10f}")
            print(f"Best y = {best.y:.10f}")
            print(f"Equation 1 = {eq1:.10e}")
            print(f"Equation 2 = {eq2:.10e}")
            print(f"Total error = {error:.10e}")
            print("======================================\n")
            break

        # =================================================
        # МІГРАЦІЯ
        # =================================================

        if generation > 0 and generation % config.migration_interval == 0:

            migrants = population[:config.migration_size]

            for pipe in send_pipes:
                pipe.send(migrants)

            migration_history.append(generation)

        # =================================================
        # ПРИЙОМ МІГРАНТІВ
        # =================================================

        for pipe in recv_pipes:

            while pipe.poll():

                incoming = pipe.recv()

                population[-len(incoming):] = incoming

                population = sorted(
                    population,
                    key=lambda ind: ind.fit,
                    reverse=True
                )

    result_queue.put({
        'island_id': island_id,
        'best_history': best_history,
        'avg_history': avg_history,
        'migration_history': migration_history,
        'iteration_times': iteration_times,
        'error_history': error_history,
        'best_individual': population[0]
    })


# =========================================================
# СТВОРЕННЯ ТОПОЛОГІЙ
# =========================================================

def build_topology(config):

    send_connections = [[] for _ in range(config.islands)]
    recv_connections = [[] for _ in range(config.islands)]

    # =====================================================
    # RING
    # =====================================================

    if config.topology == 'ring':

        for i in range(config.islands):

            sender, receiver = mp.Pipe()

            next_island = (i + 1) % config.islands

            send_connections[i].append(sender)
            recv_connections[next_island].append(receiver)

    # =====================================================
    # ALL TO ALL
    # =====================================================

    elif config.topology == 'all':

        for i in range(config.islands):

            for j in range(config.islands):

                if i != j:

                    sender, receiver = mp.Pipe()

                    send_connections[i].append(sender)
                    recv_connections[j].append(receiver)

    # =====================================================
    # STAR
    # =====================================================

    elif config.topology == 'star':

        center = 0

        for i in range(1, config.islands):

            sender1, receiver1 = mp.Pipe()
            sender2, receiver2 = mp.Pipe()

            # center -> node
            send_connections[center].append(sender1)
            recv_connections[i].append(receiver1)

            # node -> center
            send_connections[i].append(sender2)
            recv_connections[center].append(receiver2)

    return send_connections, recv_connections


# =========================================================
# RUN DGA
# =========================================================

def run_dga(config):

    total_start = time.time()

    result_queue = mp.Queue()

    processes = []

    send_connections, recv_connections = build_topology(config)

    for island_id in range(config.islands):

        process = mp.Process(
            target=evolve_island,
            args=(
                island_id,
                config,
                send_connections[island_id],
                recv_connections[island_id],
                result_queue
            )
        )

        processes.append(process)

        process.start()

    results = []

    for _ in range(config.islands):
        results.append(result_queue.get())

    for process in processes:
        process.join()

    total_end = time.time()
    total_time = total_end - total_start

    return results, total_time


# =========================================================
# ГРАФІКИ
# =========================================================

def plot_best_fitness(results, topology):

    plt.figure(figsize=(12, 6))

    for result in results:

        plt.plot(
            result['best_history'],
            label=f"Island {result['island_id']}"
        )

    plt.title(f"Best Fitness ({topology})")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.grid()
    plt.legend()

    plt.show()


def plot_average_fitness(results, topology):

    plt.figure(figsize=(12, 6))

    for result in results:

        plt.plot(
            result['avg_history'],
            label=f"Island {result['island_id']}"
        )

    plt.title(f"Average Fitness ({topology})")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")

    plt.grid()
    plt.legend()

    plt.show()


def plot_migrations(results, topology):

    plt.figure(figsize=(10, 5))

    for result in results:

        y = [result['island_id']] * len(result['migration_history'])

        plt.scatter(
            result['migration_history'],
            y,
            s=100,
            label=f"Island {result['island_id']}"
        )

    plt.title(f"Migration Events ({topology})")
    plt.xlabel("Generation")
    plt.ylabel("Island")

    plt.grid()
    plt.legend()

    plt.show()


def plot_iteration_times(results, topology):

    plt.figure(figsize=(12, 6))

    for result in results:

        plt.plot(
            result['iteration_times'],
            label=f"Island {result['island_id']}"
        )

    plt.title(f"Iteration Time ({topology})")
    plt.xlabel("Generation")
    plt.ylabel("Time, sec")

    plt.grid()
    plt.legend()

    plt.show()


def plot_error_history(results, topology):

    plt.figure(figsize=(12, 6))

    for result in results:

        plt.plot(
            result['error_history'],
            label=f"Island {result['island_id']}"
        )

    plt.title(f"Error History ({topology})")
    plt.xlabel("Generation")
    plt.ylabel("Total Error")

    plt.yscale("log")
    plt.grid()
    plt.legend()

    plt.show()


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nОберіть тип міграції:")
    print("1 - all  | всі острови обмінюються з усіма")
    print("2 - ring | міграція по кільцю")
    print("3 - star | міграція через центральний острів")

    choice = input("Ваш вибір: ")

    if choice == "1":
        topology = "all"
    elif choice == "2":
        topology = "ring"
    elif choice == "3":
        topology = "star"
    else:
        print("Невірний вибір. За замовчуванням використано ring.")
        topology = "ring"

    print("\n==============================")
    print(f"TOPOLOGY: {topology}")
    print("==============================")

    config = GAConfig(
        population_size=100,
        generations=200,
        mutation_rate=0.1,
        crossover_rate=0.8,
        migration_interval=10,
        migration_size=4,
        islands=4,

        x_min=-10,
        x_max=10,

        y_min=-10,
        y_max=10,

        topology=topology,
        target_error=1e-6
    )

    results, total_time = run_dga(config)

    best_result = max(
        results,
        key=lambda r: r['best_individual'].fit
    )

    best = best_result['best_individual']

    eq1, eq2 = equations(best.x, best.y)
    best_error = total_error(best.x, best.y)

    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")
    print(f"Best island = {best_result['island_id']}")
    print(f"Best x = {best.x:.10f}")
    print(f"Best y = {best.y:.10f}")
    print(f"Equation 1 = {eq1:.10e}")
    print(f"Equation 2 = {eq2:.10e}")
    print(f"Total error = {best_error:.10e}")
    print(f"Fitness = {best.fit:.10f}")
    print(f"Total execution time = {total_time:.6f} sec")
    print("==============================")

    for result in results:

        if len(result['iteration_times']) > 0:
            avg_iter_time = np.mean(result['iteration_times'])
            total_iter_time = np.sum(result['iteration_times'])
        else:
            avg_iter_time = 0
            total_iter_time = 0

        print(
            f"Island {result['island_id']} | "
            f"Average iteration time = {avg_iter_time:.6f} sec | "
            f"Total iteration time = {total_iter_time:.6f} sec"
        )

    plot_best_fitness(results, topology)
    plot_average_fitness(results, topology)
    plot_migrations(results, topology)
    plot_iteration_times(results, topology)
    plot_error_history(results, topology)


# =========================================================

if __name__ == '__main__':

    mp.freeze_support()

    main()