import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


# =========================================================
# СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ
# =========================================================
# Нова система:
#
# 1) sin(x) + y^2 - 1.5 = 0
# 2) x^2 + cos(y) - 2 = 0
#
# Шукаємо x, y, щоб обидва рівняння були близькі до 0
# =========================================================

def equations(x, y):
    eq1 = np.sin(x) + y**2 - 1.5
    eq2 = x**2 + np.cos(y) - 2

    return eq1, eq2


# =========================================================
# TOTAL ERROR
# =========================================================

def total_error(x, y):
    eq1, eq2 = equations(x, y)

    error = np.sqrt(eq1**2 + eq2**2)

    return error


# =========================================================
# FITNESS FUNCTION
# =========================================================

def fitness(x, y):
    error = total_error(x, y)

    return 1 / (1 + error)


# =========================================================
# ПАРАМЕТРИ CELLULAR GA
# =========================================================

@dataclass
class GAConfig:
    rows: int
    cols: int
    generations: int
    mutation_rate: float
    crossover_rate: float

    x_min: float
    x_max: float

    y_min: float
    y_max: float

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
# СТВОРЕННЯ ОСОБИНИ
# =========================================================

def create_individual(config):
    x = random.uniform(config.x_min, config.x_max)
    y = random.uniform(config.y_min, config.y_max)

    return Individual(x, y, fitness(x, y))


# =========================================================
# СТВОРЕННЯ СІТКИ ПОПУЛЯЦІЇ
# =========================================================

def create_grid_population(config):
    grid = []

    for i in range(config.rows):
        row = []

        for j in range(config.cols):
            row.append(create_individual(config))

        grid.append(row)

    return grid


# =========================================================
# ОТРИМАННЯ СУСІДІВ
# =========================================================
# Окіл фон Неймана:
# сама особина + верх + низ + ліво + право
# =========================================================

def get_neighbors(grid, i, j, config):
    neighbors = []

    positions = [
        (i, j),
        ((i - 1) % config.rows, j),
        ((i + 1) % config.rows, j),
        (i, (j - 1) % config.cols),
        (i, (j + 1) % config.cols)
    ]

    for r, c in positions:
        neighbors.append(grid[r][c])

    return neighbors


# =========================================================
# ТУРНІРНА СЕЛЕКЦІЯ СЕРЕД СУСІДІВ
# =========================================================

def tournament_selection(neighbors, k=3):
    selected = random.sample(neighbors, k)

    return max(selected, key=lambda ind: ind.fit)


# =========================================================
# СХРЕЩУВАННЯ
# =========================================================

def crossover(parent1, parent2, config):
    if random.random() < config.crossover_rate:
        alpha = random.random()

        child_x = alpha * parent1.x + (1 - alpha) * parent2.x
        child_y = alpha * parent1.y + (1 - alpha) * parent2.y

    else:
        child_x = parent1.x
        child_y = parent1.y

    return Individual(child_x, child_y, fitness(child_x, child_y))


# =========================================================
# МУТАЦІЯ
# =========================================================

def mutate(individual, config):
    if random.random() < config.mutation_rate:
        individual.x += random.gauss(0, 0.5)
        individual.y += random.gauss(0, 0.5)

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
# ПОШУК НАЙКРАЩОЇ ОСОБИНИ НА СІТЦІ
# =========================================================

def find_best(grid):
    best = grid[0][0]

    for row in grid:
        for ind in row:
            if ind.fit > best.fit:
                best = ind

    return best


# =========================================================
# СЕРЕДНІЙ FITNESS
# =========================================================

def average_fitness(grid):
    values = []

    for row in grid:
        for ind in row:
            values.append(ind.fit)

    return np.mean(values)


# =========================================================
# CELLULAR GENETIC ALGORITHM
# =========================================================

def cellular_ga(config):
    grid = create_grid_population(config)

    best_history = []
    avg_history = []
    error_history = []
    iteration_times = []

    start_time = time.time()

    for generation in range(config.generations):
        iter_start = time.time()

        new_grid = []

        for i in range(config.rows):
            new_row = []

            for j in range(config.cols):
                neighbors = get_neighbors(grid, i, j, config)

                p1 = tournament_selection(neighbors)
                p2 = tournament_selection(neighbors)

                child = crossover(p1, p2, config)
                child = mutate(child, config)

                current = grid[i][j]

                if child.fit > current.fit:
                    new_row.append(child)
                else:
                    new_row.append(current)

            new_grid.append(new_row)

        grid = new_grid

        best = find_best(grid)
        avg_fit = average_fitness(grid)
        error = total_error(best.x, best.y)

        best_history.append(best.fit)
        avg_history.append(avg_fit)
        error_history.append(error)

        iter_end = time.time()
        iteration_time = iter_end - iter_start
        iteration_times.append(iteration_time)

        eq1, eq2 = equations(best.x, best.y)

        print(
            f"Generation {generation} | "
            f"Best fitness = {best.fit:.10f} | "
            f"Error = {error:.10e} | "
            f"Time = {iteration_time:.6f} sec"
        )

        if error < config.target_error:
            print("\n================================")
            print("TARGET ACCURACY REACHED")
            print("================================")
            print(f"Generation = {generation}")
            print(f"Best x = {best.x:.10f}")
            print(f"Best y = {best.y:.10f}")
            print(f"Equation 1 = {eq1:.10e}")
            print(f"Equation 2 = {eq2:.10e}")
            print(f"Total error = {error:.10e}")
            print("================================")
            break

    total_time = time.time() - start_time

    return {
        "grid": grid,
        "best_history": best_history,
        "avg_history": avg_history,
        "error_history": error_history,
        "iteration_times": iteration_times,
        "total_time": total_time
    }


# =========================================================
# ГРАФІКИ
# =========================================================

def plot_best_fitness(best_history):
    plt.figure(figsize=(12, 6))
    plt.plot(best_history)
    plt.title("Best Fitness - Cellular GA")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()


def plot_average_fitness(avg_history):
    plt.figure(figsize=(12, 6))
    plt.plot(avg_history)
    plt.title("Average Fitness - Cellular GA")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.grid()
    plt.show()


def plot_error(error_history):
    plt.figure(figsize=(12, 6))
    plt.plot(error_history)
    plt.title("Error History - Cellular GA")
    plt.xlabel("Generation")
    plt.ylabel("Total Error")
    plt.yscale("log")
    plt.grid()
    plt.show()


def plot_iteration_times(iteration_times):
    plt.figure(figsize=(12, 6))
    plt.plot(iteration_times)
    plt.title("Iteration Time - Cellular GA")
    plt.xlabel("Generation")
    plt.ylabel("Time, sec")
    plt.grid()
    plt.show()


def plot_combined(best_history, avg_history):
    plt.figure(figsize=(12, 6))
    plt.plot(best_history, label="Best Fitness")
    plt.plot(avg_history, label="Average Fitness")
    plt.title("Cellular GA Convergence")
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
        rows=30,
        cols=30,
        generations=200,
        mutation_rate=0.1,
        crossover_rate=0.8,

        x_min=-10,
        x_max=10,

        y_min=-10,
        y_max=10,

        target_error=1e-6
    )

    print("\n================================")
    print("CELLULAR GENETIC ALGORITHM")
    print("SYSTEM OF NONLINEAR EQUATIONS")
    print("================================")

    results = cellular_ga(config)

    best = find_best(results["grid"])

    eq1, eq2 = equations(best.x, best.y)
    error = total_error(best.x, best.y)

    print("\n================================")
    print("FINAL RESULT")
    print("================================")
    print(f"Best x = {best.x:.10f}")
    print(f"Best y = {best.y:.10f}")
    print(f"Equation 1 = {eq1:.10e}")
    print(f"Equation 2 = {eq2:.10e}")
    print(f"Total error = {error:.10e}")
    print(f"Fitness = {best.fit:.10f}")
    print(f"Total execution time = {results['total_time']:.6f} sec")
    print(f"Average iteration time = {np.mean(results['iteration_times']):.6f} sec")
    print("================================")

    plot_best_fitness(results["best_history"])
    plot_average_fitness(results["avg_history"])
    plot_error(results["error_history"])
    plot_iteration_times(results["iteration_times"])
    plot_combined(
        results["best_history"],
        results["avg_history"]
    )


# =========================================================

if __name__ == "__main__":
    main()