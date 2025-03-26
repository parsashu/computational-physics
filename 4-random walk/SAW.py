import numpy as np
import matplotlib.pyplot as plt
import random


def self_avoiding_walk():
    visited = [(0, 0)]

    step_vectors = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    ]

    x = [0]
    y = [0]
    i = 1
    
    while True:
        available_steps = []
        for dx, dy in step_vectors:
            next_pos = (x[i - 1] + dx, y[i - 1] + dy)
            if next_pos not in visited:
                available_steps.append((dx, dy))

        if not available_steps:
            break

        dx, dy = random.choice(available_steps)

        x.append(x[i - 1] + dx)
        y.append(y[i - 1] + dy)

        visited.append((x[i], y[i]))
        i += 1

    return x[:i], y[:i]


x_pos, y_pos = self_avoiding_walk()

plt.figure(figsize=(10, 8))
plt.plot(x_pos, y_pos, "b-", alpha=0.5, label="Path")
plt.plot(x_pos[0], y_pos[0], "go", markersize=10, label="Start")
plt.plot(x_pos[-1], y_pos[-1], "ro", markersize=10, label="End")
plt.title("2D Self-Avoiding Walk")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True, which='both')
plt.xticks(np.arange(min(x_pos)-1, max(x_pos)+2, 1.0))
plt.yticks(np.arange(min(y_pos)-1, max(y_pos)+2, 1.0))
plt.axis("equal")
plt.legend()
plt.show()


# # Plot r^2 vs time
# time_range = np.linspace(1, 1000, 20, dtype=int)
# r2_list = []
# num_ensemble = 1000

# for time in time_range:
#     ensemble_r2 = []

#     for _ in range(num_ensemble):
#         x_pos, y_pos = self_avoiding_walk(time)
#         r2 = x_pos[-1] ** 2 + y_pos[-1] ** 2
#         ensemble_r2.append(r2)

#     avg_r2 = np.mean(ensemble_r2)
#     r2_list.append(avg_r2)


# plt.figure(figsize=(10, 6))
# plt.scatter(time_range, r2_list, c="b", s=20)
# plt.title(f"r^2 from Origin vs Time (n={num_ensemble} ensembles)")
# plt.xlabel("t")
# plt.ylabel("r^2 from Origin")
# plt.grid(True)
# plt.show()
