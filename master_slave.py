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
    eq2 = x - np.cos(y)

    return eq1, eq2


# =========================================================
# TOTAL ERROR
# =========================================================

def total_error(x, y):
    eq1, eq2 = equations(x, y)

    return np.sqrt(eq1**2 + eq2**2)


# =========================================================
# FITNESS
# =========================================================

def fitness(x, y):
    error = total_error(x, y)

    return 1 / (1 + error)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class GAConfig:
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float

    x_min: float
    x_max: float

    y_min: float
    y_max: float

    workers: int
    target_error: float


# =========================================================
# INDIVIDUAL
# =========================================================

@dataclass
class Individual:
    x: float
    y: float
    fit: float = 0.0


# =========================================================
# CREATE POPULATION
# =========================================================

def create_population(config):
    population = []

    for _ in range(config.population_size):
        x = random.uniform(config.x_min, config.x_max)
        y = random.uniform(config.y_min, config.y_max)

        population.append(Individual(x, y))

    return population


# =========================================================
# PARALLEL FITNESS EVALUATION
# =========================================================

def slave_fitness(individual):
    individual.fit = fitness(individual.x, individual.y)

    return individual


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

    return Individual(child_x, child_y)


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

    return individual


# =========================================================
# MASTER-SLAVE GENETIC ALGORITHM
# =========================================================

def master_slave_ga(config):
    pool = mp.Pool(processes=config.workers)

    population = create_population(config)
    population = pool.map(slave_fitness, population)

    best_history = []
    avg_history = []
    error_history = []
    iteration_times = []

    start_time = time.time()

    for generation in range(config.generations):
        iter_start = time.time()

        new_population = []

        while len(new_population) < config.population_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)

            child = crossover(p1, p2, config)
            child = mutate(child, config)

            new_population.append(child)

        population = pool.map(slave_fitness, new_population)

        population = sorted(
            population,
            key=lambda ind: ind.fit,
            reverse=True
        )

        best = population[0]

        error = total_error(best.x, best.y)

        best_history.append(best.fit)

        avg_history.append(
            np.mean([ind.fit for ind in population])
        )

        error_history.append(error)

        iter_end = time.time()
        iteration_time = iter_end - iter_start
        iteration_times.append(iteration_time)

        print(
            f"Generation {generation} | "
            f"Best fitness = {best.fit:.10f} | "
            f"Error = {error:.10e} | "
            f"Time = {iteration_time:.6f} sec"
        )

        # =================================================
        # STOPPING CRITERION BY ACCURACY
        # =================================================

        if error < config.target_error:
            print("\n======================================")
            print("TARGET ACCURACY REACHED")
            print("======================================")
            print(f"Generation = {generation}")
            print(f"Best x = {best.x:.10f}")
            print(f"Best y = {best.y:.10f}")

            eq1, eq2 = equations(best.x, best.y)

            print(f"Equation 1 = {eq1:.10e}")
            print(f"Equation 2 = {eq2:.10e}")
            print(f"Total error = {error:.10e}")
            print("======================================\n")
            break

    end_time = time.time()
    execution_time = end_time - start_time

    pool.close()
    pool.join()

    return {
        'population': population,
        'best_history': best_history,
        'avg_history': avg_history,
        'error_history': error_history,
        'iteration_times': iteration_times,
        'execution_time': execution_time
    }


# =========================================================
# PLOTS
# =========================================================

def plot_best_fitness(best_history):
    plt.figure(figsize=(12, 6))
    plt.plot(best_history)
    plt.title("Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()


def plot_average_fitness(avg_history):
    plt.figure(figsize=(12, 6))
    plt.plot(avg_history)
    plt.title("Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.grid()
    plt.show()


def plot_error_history(error_history):
    plt.figure(figsize=(12, 6))
    plt.plot(error_history)
    plt.title("Error History")
    plt.xlabel("Generation")
    plt.ylabel("Total Error")
    plt.yscale("log")
    plt.grid()
    plt.show()


def plot_iteration_times(iteration_times):
    plt.figure(figsize=(12, 6))
    plt.plot(iteration_times)
    plt.title("Iteration Time")
    plt.xlabel("Generation")
    plt.ylabel("Time, sec")
    plt.grid()
    plt.show()


def plot_combined(best_history, avg_history):
    plt.figure(figsize=(12, 6))
    plt.plot(best_history, label='Best Fitness')
    plt.plot(avg_history, label='Average Fitness')
    plt.title("GA Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid()
    plt.show()


# =========================================================
# MAIN
# =========================================================

def main():
    config = GAConfig(
        population_size=100,
        generations=200,
        mutation_rate=0.1,
        crossover_rate=0.8,

        x_min=-10,
        x_max=10,

        y_min=-10,
        y_max=10,

        workers=4,
        target_error=1e-6
    )

    print("\n======================================")
    print("MASTER-SLAVE GA FOR SYSTEM OF NONLINEAR EQUATIONS")
    print("======================================")

    results = master_slave_ga(config)

    best = results['population'][0]

    eq1, eq2 = equations(best.x, best.y)
    error = total_error(best.x, best.y)

    print("\n======================================")
    print("FINAL RESULT")
    print("======================================")
    print(f"Best x = {best.x:.10f}")
    print(f"Best y = {best.y:.10f}")
    print(f"Equation 1 = {eq1:.10e}")
    print(f"Equation 2 = {eq2:.10e}")
    print(f"Total error = {error:.10e}")
    print(f"Fitness = {best.fit:.10f}")
    print(f"Execution time = {results['execution_time']:.6f} sec")
    print(f"Average iteration time = {np.mean(results['iteration_times']):.6f} sec")
    print("======================================")

    plot_best_fitness(results['best_history'])
    plot_average_fitness(results['avg_history'])
    plot_combined(
        results['best_history'],
        results['avg_history']
    )
    plot_error_history(results['error_history'])
    plot_iteration_times(results['iteration_times'])


# =========================================================

if __name__ == '__main__':
    mp.freeze_support()
    main()