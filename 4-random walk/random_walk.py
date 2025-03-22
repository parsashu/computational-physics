"""
This script simulates a one-dimensional random walk and visualizes the path
taken by a point starting at the origin and moving in random directions in 1D space.
"""

import numpy as np
import matplotlib.pyplot as plt

p = 0.5
l = 1
T = 1
q = 1 - p
num_steps = 1000
ensamble = 500


def single_random_walk(num_steps):
    """Simulates a single random walk."""
    position = 0
    path = [0]

    for _ in range(num_steps):
        step_direction = np.random.choice([-l, l], p=[q, p])
        position += step_direction
        path.append(position)

    return path


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


times = [step * T for step in range(num_steps + 1)]
sim_mean, sim_variance = ensemble_average(num_steps, ensamble)
theo_mean = theorical_mean(times)
theo_variance = theoretical_variance(times)
single_path = single_random_walk(num_steps)

plt.figure(figsize=(10, 6))
plt.plot(times, single_path, alpha=0.5, label="Example Random Walk")
plt.plot(
    times,
    sim_mean,
    "g-",
    linewidth=2,
    label=f"Simulated Mean (ensamble = {ensamble})",
)
plt.plot(
    times,
    theo_mean,
    "r--",
    linewidth=2,
    label="Theoretical Mean: $\\langle x(t) \\rangle = \\frac{l}{\\tau}(p-q)t$",
)
plt.title(f"One-dimensional Random Walk\np={p}, l={l}, T={T}, N={num_steps}")
plt.xlabel("Time")
plt.ylabel("Position")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(
    times,
    sim_variance,
    "g-",
    linewidth=2,
    label=f"Simulated Variance (ensamble = {ensamble})",
)
plt.plot(
    times,
    theo_variance,
    "r--",
    linewidth=2,
    label="Theoretical Variance: $\\sigma^2(t) = 4l^2pq\\frac{t}{\\tau}$",
)
plt.title(
    f"Variance of One-dimensional Random Walk\np={p}, l={l}, T={T}, N={num_steps}"
)
plt.xlabel("Time")
plt.ylabel("Variance")
plt.grid(True)
plt.legend()
plt.show()
