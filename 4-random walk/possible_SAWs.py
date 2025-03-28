import numpy as np
import matplotlib.pyplot as plt
import random

N = 5


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

    while len(x) <= N:
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

    length = len(x) - 1
    return x[:i], y[:i], length


x_pos, y_pos, length = self_avoiding_walk()

plt.figure(figsize=(10, 8))
plt.plot(x_pos, y_pos, "b-", alpha=0.5, label="Path")
plt.plot(x_pos[0], y_pos[0], "go", markersize=10, label="Start")
plt.plot(x_pos[-1], y_pos[-1], "ro", markersize=10, label="End")
plt.title("2D Self-Avoiding Walk")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True, which="both")
plt.xticks(np.arange(min(x_pos) - 1, max(x_pos) + 2, 1.0))
plt.yticks(np.arange(min(y_pos) - 1, max(y_pos) + 2, 1.0))
plt.axis("equal")
plt.legend()
plt.show()


# # Plot length distribution
# length_list = []
# num_runs = 10000

# for _ in range(num_runs):
#     _, _, length = self_avoiding_walk()
#     length_list.append(length)


# plt.figure(figsize=(10, 6))
# plt.hist(length_list, bins=np.arange(0, max(length_list) + 20, 10), edgecolor='black')
# plt.title(f"Length Distribution (n={num_runs} runs)")
# plt.xlabel("Length")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()
