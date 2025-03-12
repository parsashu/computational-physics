import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

L = 50
p = 0.5
int_max = 1000
color = 2
color_min = 0

grid = np.zeros((L, L))
grid[:, 0] = 1
grid[:, -1] = int_max

random_values = np.random.random((L, L))
color_change_list = []


def get_neighbors(grid, i, j):
    neighbors = {}

    # Up
    if i > 0:
        neighbors[i - 1, j] = grid[i - 1, j]
    # Down
    if i < grid.shape[0] - 1:
        neighbors[i + 1, j] = grid[i + 1, j]
    # Left
    if j > 0:
        neighbors[i, j - 1] = grid[i, j - 1]
    # Right
    if j < grid.shape[1] - 1:
        neighbors[i, j + 1] = grid[i, j + 1]
    return neighbors


def are_all_neighbors_zero(neighbors):
    for ni, nj in neighbors:
        if grid[ni, nj] != 0:
            return False
    return True


for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if random_values[i, j] < p:
            grid[i, j] = color
            color += 1

            neighbors = get_neighbors(grid, i, j)
            if not are_all_neighbors_zero(neighbors):
                non_zero_values = [value for value in neighbors.values() if value != 0]
                if len(non_zero_values) == 1:
                    grid[i, j] = non_zero_values[0]
                else:
                    color_min = min(non_zero_values)
                    grid[i, j] = color_min
                    color_change_list.append(color_min)
                    # Last box


plt.figure(figsize=(6, 6))
base_cmap = cm.get_cmap("plasma", 101)
colors = [base_cmap(i) for i in range(101)]
colors[0] = (0.3, 0.3, 0.3, 1.0)
colors[1] = (0.3, 0.3, 0.3, 1.0)
cmap = ListedColormap(colors)
plt.imshow(grid, cmap=cmap, vmin=0, vmax=color)
plt.colorbar(label="Value")
plt.grid(False)
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.1)
plt.xticks([])
plt.yticks([])
plt.title(f"Percolation Model with Values 1-{int_max} (L={L}, p={p})")
plt.show()
