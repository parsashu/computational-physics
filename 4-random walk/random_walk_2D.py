import numpy as np
import matplotlib.pyplot as plt
import random

l = 1
T = 1
D = l**2 / (2 * T)


def random_walk_2D(steps):
    step_vectors = [
        (l, 0),
        (0, l),
        (-l, 0),
        (0, -l),
    ]

    x_positions = np.zeros(steps + 1)
    y_positions = np.zeros(steps + 1)

    for i in range(1, steps + 1):
        dx, dy = random.choice(step_vectors)

        x_positions[i] = x_positions[i - 1] + dx
        y_positions[i] = y_positions[i - 1] + dy

    return x_positions, y_positions


num_steps = 1000
x_pos, y_pos = random_walk_2D(num_steps)

plt.figure(figsize=(10, 8))
plt.plot(x_pos, y_pos, "b-", alpha=0.5, label="Path")
plt.plot(x_pos[0], y_pos[0], "go", markersize=10, label="Start")
plt.plot(x_pos[-1], y_pos[-1], "ro", markersize=10, label="End")
plt.title("2D Random Walk")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()


time_range = np.linspace(1, 1000, 100, dtype=int)
r2_list = []
num_ensemble = 100

for time in time_range:
    ensemble_r2 = []

    for _ in range(num_ensemble):
        x_pos, y_pos = random_walk_2D(time)
        r2 = x_pos[-1] ** 2 + y_pos[-1] ** 2
        ensemble_r2.append(r2)

    avg_r2 = np.mean(ensemble_r2)
    r2_list.append(avg_r2)


def theoretical_r2(time):
    return time * 4 * D


plt.figure(figsize=(10, 6))
plt.scatter(time_range, r2_list, c="b", s=20)
plt.plot(time_range, theoretical_r2(time_range), "r-", label="Theoretical")
plt.title(f"r^2 from Origin vs Time (n={num_ensemble} ensembles)")
plt.xlabel("t")
plt.ylabel("r^2 from Origin")
plt.grid(True)
plt.show()
