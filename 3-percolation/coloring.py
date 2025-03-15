import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time


L = 100
p = 0.59
int_max = 10000
color = 2
color_min = 0
# np.random.seed(12)

grid = np.zeros((L, L))
grid[:, 0] = 1
grid[:, -1] = int_max

random_values = np.random.random((L, L))


def get_neighbors(grid, i, j):
    """Get the up, down, left and right neighbors of the cell"""
    neighbors = {}

    # Up
    if i > 0 and grid[i - 1, j] != 0:
        neighbors[i - 1, j] = grid[i - 1, j]

    # Down
    if i < grid.shape[0] - 1 and grid[i + 1, j] != 0:
        neighbors[i + 1, j] = grid[i + 1, j]

    # Left
    if j > 0 and grid[i, j - 1] != 0:
        neighbors[i, j - 1] = grid[i, j - 1]

    # Right
    if j < grid.shape[1] - 1 and grid[i, j + 1] != 0:
        neighbors[i, j + 1] = grid[i, j + 1]
    return neighbors

def is_percolating():
    """Check if the cluster percolates"""
    if grid[0, -1] == 1:
        return True
    return False


start_time = time.time()

def coloring(grid, random_values, p):
    global color

    for i in range(L):
        for j in range(L):
            color_change_list = []

            if random_values[i, j] < p and grid[i, j] == 0:
                grid[i, j] = color
                color += 1

                neighbors = get_neighbors(grid, i, j)
                if len(neighbors) == 1:
                    grid[i, j] = list(neighbors.values())[0]

                elif len(neighbors) > 1:
                    color_min = min(list(neighbors.values()))
                    grid[i, j] = color_min
                    for value in list(neighbors.values()):
                        if value != color_min:
                            color_change_list.append(value)

                    for i_ in range(L):
                        for j_ in range(L):
                            if grid[i_, j_] in color_change_list:
                                grid[i_, j_] = color_min
    return is_percolating()


def plot():
    percolates = coloring(grid, random_values, p)
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")

    plt.figure(figsize=(6, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, int(color)))
    colors[0] = (0.3, 0.3, 0.3, 1.0)
    colors[-1] = (0, 0, 0, 1.0)
    colors[1] = (0, 0, 0, 0)
    cmap = ListedColormap(colors)

    plt.imshow(grid, cmap=cmap, vmin=0, vmax=color - 1)
    plt.colorbar(label="Cluster ID")
    plt.grid(False)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.1)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Percolation Clusters (L={L}, p={p}, Percolates={percolates})")
    plt.show()
    
# plot()
    
