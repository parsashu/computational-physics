import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time


# np.random.seed(12)


def get_up_left_neighbors(grid, i, j):
    """Get the up and left neighbors of the cell"""
    neighbors_k = []

    # Up
    if i > 0 and grid[i - 1, j] != 0:
        neighbors_k.append(grid[i - 1, j])

    # Left
    if j > 0 and grid[i, j - 1] != 0:
        neighbors_k.append(grid[i, j - 1])

    return neighbors_k


def root(label):
    """Find the root of the label with path compression"""
    global L

    if label not in L:
        L[label] = label
        return label

    if L[label] != label:
        L[label] = root(L[label])
    return L[label]


def is_percolating(grid):
    """Check if the cluster percolates"""
    for cell in grid[:, -1]:
        if cell != 0:
            if root(int(cell)) == root(1):
                return True
    return False


def merge_clusters(grid):
    """Plot the clusters with the same root as same color"""
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] > 0:
                grid[i, j] = root(int(grid[i, j]))
    return grid


start_time = time.time()


def hoshen_kopelman(length, random_values, p):
    global k, L, S

    k = 2
    L = {1: 1}
    S = {1: length}

    grid = np.zeros((length, length))
    grid[:, 0] = 1

    for j in range(grid.shape[1]):
        for i in range(grid.shape[0]):
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
                        L[root(k_up)] = root(k_left)
                        S[k_left] += S[k_up] + 1
                        S[k_up] = 0
                    else:
                        grid[i, j] = k_left
                        S[k_left] += 1

    return merge_clusters(grid), is_percolating(grid)


def is_connected_to_infinite_cluster_hoshen(grid):
    """Calculate if a random point is connected to the infinite cluster"""
    while True:
        i = np.random.randint(0, grid.shape[0])
        j = np.random.randint(0, grid.shape[1])
        if grid[i, j] != 0:
            break

    if root(int(grid[i, j])) == root(1):
        return True
    return False


def plot(length, p):
    random_values = np.random.random((length, length))
    grid, percolates = hoshen_kopelman(length, random_values, p)

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.3f} seconds")

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


# plot(length=100, p=0.59)
