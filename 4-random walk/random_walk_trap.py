import numpy as np
import matplotlib.pyplot as plt

p = 0.5
l = 1
T = 1
q = 1 - p
trap_x = 10
ensamble = 500


def single_random_walk():
    """Simulates a single random walk."""
    position = 0
    path = [0]

    while position <= trap_x and position >= -trap_x:
        step_direction = np.random.choice([-l, l], p=[q, p])
        position += step_direction
        path.append(position)

    times = [step * T for step in range(len(path))]
    return path, times


def ensemble_average(num_steps, ensamble):
    """Calculates the ensemble average over multiple trials."""
    all_paths = np.zeros((ensamble, num_steps + 1))

    for trial in range(ensamble):
        all_paths[trial, :] = single_random_walk(num_steps)

    mean_positions = np.mean(all_paths, axis=0)
    mean_positions_square = np.mean(all_paths**2, axis=0)
    variance = mean_positions_square - mean_positions**2
    return mean_positions, variance


def theorical_mean(times):
    return (l / T) * (p - q) * np.array(times)


def theoretical_variance(times):
    return 4 * l**2 * (p * q) * (1 / T) * np.array(times)


single_path, times = single_random_walk()


plt.figure(figsize=(10, 6))
plt.plot(times, single_path, alpha=0.5, label="Example Random Walk")
plt.title(f"One-dimensional Random Walk\np={p}, l={l}, T={T}")
plt.xlabel("Time")
plt.ylabel("Position")
plt.grid(True)
plt.legend()
plt.xticks(np.linspace(min(times), max(times), 10))
plt.show()
