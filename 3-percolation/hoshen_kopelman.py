import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

length = 50
p = 0.6
k = 2
L = {}
S = {1: length}
np.random.seed(12)

grid = np.zeros((length, length))
grid[:, 0] = 1

random_values = np.random.random((length, length))


def get_up_left_neighbors(grid, i, j):
    neighbors_k = []

    # Up
    if i > 0 and grid[i - 1, j] != 0:
        neighbors_k.append(grid[i - 1, j])

    # Left
    if j > 0 and grid[i, j - 1] != 0:
        neighbors_k.append(grid[i, j - 1])

    return neighbors_k


for j in range(length):
    for i in range(length):
        if random_values[i, j] < p and grid[i, j] == 0:
            neighbors_k = get_up_left_neighbors(grid, i, j)

            if len(neighbors_k) == 0:
                grid[i, j] = k
                L[k] = k
                S[k] = 1
                k += 1

            elif len(neighbors_k) == 1:
                k_neighbor = int(neighbors_k[0])
                grid[i, j] = k_neighbor
                S[k_neighbor] += 1

            else:
                k_up = int(neighbors_k[0])
                k_left = int(neighbors_k[1])
                if k_up != k_left:
                    grid[i, j] = k_left
                    L[k_up] = k_left
                    S[k_left] += S[k_up] + 1
                    S[k_up] = 0
                else:
                    grid[i, j] = k_left
                    S[k_left] += 1


def root(label):
    if label not in L:
        L[label] = label

    while label != L[label]:
        label = L[label]
    return label


for i in range(length):
    for j in range(length):
        if grid[i, j] > 0:
            grid[i, j] = root(int(grid[i, j]))


def is_percolating():
    left_root = root(1)

    for cell in grid[:, -1]:
        if cell != 0:
            if L[int(cell)] == left_root:
                return True
    return False


percolates = is_percolating()

plt.figure(figsize=(6, 6))
colors = plt.cm.rainbow(np.linspace(0, 1, int(k)))
colors[0] = (0.3, 0.3, 0.3, 1.0)
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
plt.title(f"Percolation Clusters (L={length}, p={p}, Percolates={percolates})")
plt.show()
