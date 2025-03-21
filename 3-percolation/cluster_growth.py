import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_neighbors(grid, i, j):
    """Get the up, down, left and right neighbors of the cell"""
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


def cluster_growth(length, p):
    grid = np.zeros((length, length))
    grid[length // 2, length // 2] = 1

    cells_to_check = [(length // 2, length // 2)]

    while cells_to_check:
        cell = cells_to_check.pop()
        neighbors = get_neighbors(grid, cell[0], cell[1])

        for key in neighbors:
            if neighbors[key] == 0:
                random_value = np.random.random()

                if random_value < p:
                    grid[key] = 1
                    cells_to_check.append(key)
                else:
                    grid[key] = -1

    return grid


def size_of_cluster(grid):
    """Calculate the size of the cluster"""
    return np.sum(grid == 1)


def center_of_mass(grid):
    """Calculate the center of mass"""
    coords = np.where(grid == 1)
    return np.array([np.mean(coords[0]), np.mean(coords[1])])


def Xi(grid):
    """Calculate the radius of gyration"""
    Rg = 0
    coords = np.where(grid == 1)

    for i in range(len(coords[0])):
        Rg += (
            np.linalg.norm(
                np.array([coords[0][i], coords[1][i]]) - center_of_mass(grid)
            )
            ** 2
        )

    Rg = (Rg / size_of_cluster(grid)) ** 0.5
    return Rg


def plot_grid(length, p):
    grid = cluster_growth(length, p)
    S = size_of_cluster(grid)
    xi = Xi(grid)

    plt.figure(figsize=(6, 6))

    colors = [
        (0.3, 0.3, 0.3, 1.0),
        (0.3, 0.3, 0.3, 1.0),
        (0, 0, 0.5, 1.0),
    ]
    cmap = ListedColormap(colors)

    plt.imshow(grid, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label="Cluster ID")
    plt.grid(False)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, length, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, length, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.1)
    plt.xticks([])
    plt.yticks([])
    plt.title(
        f"Diffusion Limited Aggregation (L={length}, p={p})\nS={S:.0f}, ξ={xi:.2f}"
    )
    plt.show()


def plot_xi_vs_s(length, ensemble):
    p_list = [0.5, 0.55, 0.59]
    log_s = []
    log_xi = []

    for p in p_list:
        s = []
        xi = []

        for _ in range(ensemble):
            grid = cluster_growth(length, p)
            s.append(size_of_cluster(grid))
            xi.append(Xi(grid))

        log_s.append(np.log(np.mean(s)))
        log_xi.append(np.log(np.mean(xi)))

    coeffs = np.polyfit(log_s, log_xi, 1)
    a, b = coeffs[0], coeffs[1]
    fit_line = [a * x + b for x in log_s]

    plt.figure(figsize=(10, 6))
    plt.plot(log_s, log_xi, "o", label="Cluster Growth")
    plt.plot(log_s, fit_line, "-", label=f"Fit: ξ ~ S^{a:.2f}")
    plt.xlabel("log(S)")
    plt.ylabel("log(ξ)")
    plt.title(f"Radius of Gyration vs Cluster Size\na={a:.2f}, b={b:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()


# np.random.seed(12)
plot_grid(length=300, p=0.59)
plot_xi_vs_s(length=200, ensemble=10)
