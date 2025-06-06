import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


N = 60


def get_neighbors(grid, i, j):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N:
                neighbors.append(grid[ni, nj])
    return neighbors


def is_satisfied(grid, i, j, Bm):
    neighbors = get_neighbors(grid, i, j)
    B = np.sum(neighbors == grid[i, j]) / len(neighbors)
    return B >= Bm


def move_agent(grid, new_grid, i, j, p_rand_move):
    """
    Move the agent to the closest empty cell or a random empty cell with probability p_rand_move
    """
    empty_cells = np.where((grid == 0) & (new_grid == 0))
    empty_cells = list(zip(empty_cells[0], empty_cells[1]))

    if not empty_cells:
        return

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


def schelling_model(rho, Bm, p_rand_move=1, max_iters=500):
    grid = np.random.choice(
        [-1, 0, 1], size=(N, N), p=[(1 - rho) / 2, rho, (1 - rho) / 2]
    )

    stable = False
    n_moves = 0

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
                move_agent(grid, new_grid, i, j, p_rand_move)
                current_n_moves += 1
        grid = new_grid

        # No agent moved
        if current_n_moves == 0:
            stable = True

        n_moves += current_n_moves

    return n_moves


# # n_moves vs Bm
# rho = 0.1
# Bms = np.linspace(0.5, 1, 10)
# n_moves = [
#     schelling_model(rho=rho, Bm=Bm)
#     for Bm in tqdm(Bms, desc="Computing moves for different Bm values")
# ]

# plt.figure(figsize=(10, 6))
# plt.plot(Bms, n_moves, "b-", linewidth=2)
# plt.xlabel("Bm (Similarity Threshold)", fontsize=12)
# plt.ylabel("Number of Moves", fontsize=12)
# plt.title(f"Number of Moves vs Similarity Threshold, rho = {rho}", fontsize=14)
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.show()


# n_moves vs rho
rhos = np.linspace(0, 0.8, 50)
Bm = 0.5
n_moves = [
    schelling_model(rho=rho, Bm=Bm)
    for rho in tqdm(rhos, desc="Computing moves for different rho values")
]

plt.figure(figsize=(10, 6))
plt.plot(rhos, n_moves, "b-", linewidth=2)
plt.xlabel("rho (Density)", fontsize=12)
plt.ylabel("Number of Moves", fontsize=12)
plt.title(f"Number of Moves vs Density, Bm = {Bm}", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
