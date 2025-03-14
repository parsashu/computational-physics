import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

length = 10
p = 0.5
k = 2
L = {}
# np.random.seed(12)

grid = np.zeros((length, length))
grid[:, 0] = 1

random_values = np.random.random((length, length))


def get_up_left_neighbors(grid, i, j):
    neighbors_coords = []
    neighbors_colors = []

    # Up
    if i > 0 and grid[i - 1, j] != 0:
        neighbors_coords.append((i - 1, j))
        neighbors_colors.append(grid[i - 1, j])

    # Left
    if j > 0 and grid[i, j - 1] != 0:
        neighbors_coords.append((i, j - 1))
        neighbors_colors.append(grid[i, j - 1])

    return neighbors_coords, neighbors_colors


for j in range(length):
    for i in range(length):
        if random_values[i, j] < p and grid[i, j] == 0:
            neighbors_coords, neighbors_colors = get_up_left_neighbors(grid, i, j)

            if len(neighbors_coords) == 0:
                grid[i, j] = k
                L[k] = k
                k += 1

            elif len(neighbors_coords) == 1:
                grid[i, j] = neighbors_colors[0]

            else:
                k_up = neighbors_colors[0]
                k_left = neighbors_colors[1]
                grid[i, j] = k_left
                L[k_up] = k_left


plt.figure(figsize=(6, 6))
colors = plt.cm.rainbow(np.linspace(0, 1, int(k)))
colors[0] = (0.3, 0.3, 0.3, 1.0)
colors[-1] = (0, 0, 0, 1.0)
colors[1] = (0, 0, 0, 0)
cmap = ListedColormap(colors)

plt.imshow(grid, cmap=cmap, vmin=0, vmax=k - 1)
plt.colorbar(label="Cluster ID")
plt.grid(False)
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, length, 1), minor=True)
ax.set_yticks(np.arange(-0.5, length, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.1)
plt.xticks([])
plt.yticks([])
plt.title(f"Percolation Clusters (L={length}, p={p})")
plt.show()
