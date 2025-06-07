import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import multiprocessing as mp
from functools import partial
import os

# np.random.seed(42)
N = 60


def get_neighbors(grid, i, j):
    neighbors_list = []

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N:
                neighbors_list.append(grid[ni, nj])
    return neighbors_list


def get_neighbors_color(grid, color_grid, i, j):
    neighbors = {}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for di, dj in directions:
        ni, nj = i + di, j + dj
        if (
            0 <= ni < N
            and 0 <= nj < N
            and grid[ni, nj] == grid[i, j]
            and color_grid[ni, nj] != 0
        ):
            neighbors[ni, nj] = color_grid[ni, nj]
    return neighbors


def is_satisfied(grid, i, j, Bm):
    neighbors = get_neighbors(grid, i, j)
    B = np.sum(neighbors == grid[i, j]) / len(neighbors)
    return B >= Bm


def move_agent(grid, new_grid, i, j, p_rand_move):
    """
    Move the agent to the closest empty cell or a random empty cell with probability p_rand_move
    """
    moved = False
    empty_cells = np.where((grid == 0) & (new_grid == 0))
    empty_cells = list(zip(empty_cells[0], empty_cells[1]))

    if not empty_cells:
        return moved

    # Random move
    if np.random.rand() < p_rand_move:
        random_cell = empty_cells[np.random.randint(len(empty_cells))]
        new_grid[random_cell[0], random_cell[1]] = grid[i, j]
        new_grid[i, j] = 0

    # Move to the closest empty cell
    else:
        distances = []
        for ei, ej in empty_cells:
            dist = np.sqrt((i - ei) ** 2 + (j - ej) ** 2)
            distances.append((dist, ei, ej))

        distances.sort()

        # Find all cells with the minimum distance
        min_dist = distances[0][0]
        closest_cells = [(ei, ej) for dist, ei, ej in distances if dist == min_dist]

        closest_i, closest_j = closest_cells[np.random.randint(len(closest_cells))]

        new_grid[closest_i, closest_j] = grid[i, j]
        new_grid[i, j] = 0
    moved = True
    return moved


# Coloring Algorithm
def coloring(grid):
    color = 2
    color_min = 0
    color_grid = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            color_change_list = []

            if grid[i, j] != 0 and color_grid[i, j] == 0:
                color_grid[i, j] = color
                color += 1

                neighbors = get_neighbors_color(grid, color_grid, i, j)
                if len(neighbors) == 1:
                    color_grid[i, j] = list(neighbors.values())[0]

                elif len(neighbors) > 1:
                    color_min = min(list(neighbors.values()))
                    color_grid[i, j] = color_min
                    for value in list(neighbors.values()):
                        if value != color_min:
                            color_change_list.append(value)

                    for i_ in range(N):
                        for j_ in range(N):
                            if color_grid[i_, j_] in color_change_list:
                                color_grid[i_, j_] = color_min
    return color_grid


def cluster_sizes(color_grid):
    cluster_sizes = {}
    for i in range(N):
        for j in range(N):
            if color_grid[i, j] != 0:
                cluster_sizes[color_grid[i, j]] = (
                    cluster_sizes.get(color_grid[i, j], 0) + 1
                )
    return cluster_sizes


def segregation_coeficient(color_grid, rho):
    n_c = cluster_sizes(color_grid)
    n_c = np.array(list(n_c.values()))
    s = 2 * np.sum(n_c**2) / (N**2 * (1 - rho)) ** 2
    return s


def schelling_model(rho, Bm, p_rand_move=1, max_iters=3000):
    grid = np.random.choice(
        [-1, 0, 1], size=(N, N), p=[(1 - rho) / 2, rho, (1 - rho) / 2]
    )
    stable = False

    for _ in range(max_iters):
        if stable:
            break

        current_n_moves = 0
        new_grid = np.copy(grid)

        # Shuffle the cells order
        positions = [(i, j) for i in range(N) for j in range(N)]
        np.random.shuffle(positions)

        for i, j in positions:
            if grid[i, j] != 0 and not is_satisfied(grid, i, j, Bm):
                moved = move_agent(grid, new_grid, i, j, p_rand_move)
                if moved:
                    current_n_moves += 1
        grid = new_grid

        # No agent moved
        if current_n_moves == 0:
            stable = True

    color_grid = coloring(grid)
    s = segregation_coeficient(color_grid, rho)
    return grid, color_grid, s


# Grid and color grid
rho = 0.01
Bm = 0.625
grid, color_grid, s = schelling_model(rho=rho, Bm=Bm)

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121)
im1 = ax1.imshow(grid, cmap="bwr", interpolation="none", vmin=-1, vmax=1, alpha=0.8)
ax1.set_xticks(np.arange(-0.5, N, 1), minor=True)
ax1.set_yticks(np.arange(-0.5, N, 1), minor=True)
ax1.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax1.set_title(f"Schelling Model\nBm = {Bm}, rho = {rho}", fontsize=16, pad=20)

ax2 = fig.add_subplot(122)
max_color = int(color_grid.max())
colors = np.random.rand(max_color, 4)
colors[0] = (1, 1, 1, 1.0)
colors[1] = (0, 0, 0, 0)
cmap = ListedColormap(colors)
im2 = ax2.imshow(color_grid, cmap=cmap, vmin=0, vmax=max_color - 1)
ax2.set_title("Color Grid (Clusters)")
ax2.set_xticks([])
ax2.set_yticks([])
plt.tight_layout()
plt.show()

# s vs Bm for different rhos
rho_list = [0.01, 0.02, 0.12, 0.18, 0.26]
Bms = np.linspace(0.2, 0.8, 20)
n_ensemble = 4


def run_single_ensemble(args):
    rho, ensemble_idx, Bms, position = args
    s_values = []

    for Bm in tqdm(
        Bms, desc=f"ρ={rho:.2f} ens={ensemble_idx}", position=position, leave=True
    ):
        _, _, s = schelling_model(rho=rho, Bm=Bm)
        s_values.append(s)

    return rho, ensemble_idx, s_values


def init_worker():
    pass


if __name__ == "__main__":
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass 

    plt.figure(figsize=(10, 6))

    args_list = []
    position = 0
    for rho in rho_list:
        for ensemble_idx in range(n_ensemble):
            args_list.append((rho, ensemble_idx, Bms, position))
            position += 1

    total_workers = len(rho_list) * n_ensemble
    print(f"Starting {total_workers} parallel workers...")

    with mp.Pool(
        processes=min(mp.cpu_count(), total_workers), initializer=init_worker
    ) as pool:
        raw_results = pool.map(run_single_ensemble, args_list)

    results_dict = {}
    for rho, ensemble_idx, s_values in raw_results:
        if rho not in results_dict:
            results_dict[rho] = []
        results_dict[rho].append(s_values)

    results = []
    for rho in rho_list:
        s_ensemble = np.array(results_dict[rho])
        s_mean = np.mean(s_ensemble, axis=0)
        results.append((rho, s_mean))

    for rho, s_mean in results:
        plt.plot(Bms, s_mean, linewidth=2, label=f"ρ = {rho}")

    plt.xlabel("Bm (Similarity Threshold)", fontsize=12)
    plt.ylabel("Segregation Coefficient", fontsize=12)
    plt.title(
        f"Segregation Coefficient vs Similarity Threshold\n{n_ensemble} ensemble runs",
        fontsize=14,
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
