import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)


def random_walk_2D(steps, step_size=1):
    step_vectors = [
        (step_size, 0),
        (0, step_size),
        (-step_size, 0),
        (0, -step_size),
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


num_steps_range = np.linspace(1, 1000, 100, dtype=int)
distances = []
num_ensemble = 1000

for num_step in num_steps_range:
    ensemble_distances = []

    for _ in range(num_ensemble):
        x_pos, y_pos = random_walk_2D(num_step)
        r = np.sqrt(x_pos[-1] ** 2 + y_pos[-1] ** 2)
        ensemble_distances.append(r)
    avg_distance = np.mean(ensemble_distances)
    distances.append(avg_distance)

plt.figure(figsize=(10, 6))
plt.plot(num_steps_range, distances, "b-", alpha=0.7)
plt.scatter(num_steps_range, distances, c="r", s=20, alpha=0.5)
plt.title("Distance from Origin vs Number of Steps")
plt.xlabel("Number of Steps")
plt.ylabel("Distance from Origin")
plt.grid(True)
plt.show()
