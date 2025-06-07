import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


N = 60
Bm = 0.625
rho = 0.01
p_rand_move = 1
iters = 1500
np.random.seed(42)

grid = np.random.choice([-1, 0, 1], size=(N, N), p=[(1 - rho) / 2, rho, (1 - rho) / 2])


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


def is_satisfied(grid, i, j):
    neighbors = get_neighbors(grid, i, j)
    B = np.sum(neighbors == grid[i, j]) / len(neighbors)
    return B >= Bm


def move_agent(grid, new_grid, i, j):
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


def update(frame):
    """Main function"""
    global grid

    if frame > 0:
        new_grid = np.copy(grid)

        # Shuffle the cells order
        positions = [(i, j) for i in range(N) for j in range(N)]
        np.random.shuffle(positions)

        for i, j in positions:
            if grid[i, j] != 0 and not is_satisfied(grid, i, j):
                move_agent(grid, new_grid, i, j)
        grid = new_grid

    ax.set_title(
        f"Schelling Model Bm = {Bm}, rho = {rho} - Frame: {frame}", fontsize=16, pad=20
    )

    im.set_array(grid)
    return [im]


fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(grid, cmap="bwr", interpolation="none", vmin=-1, vmax=1, alpha=0.8)
ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title(f"Schelling Model Bm = {Bm}, rho = {rho} - Frame: 0", fontsize=16, pad=20)
plt.tight_layout()
ani = animation.FuncAnimation(fig, update, frames=iters, interval=100, blit=False)
ani.save("Project/full_seperation.gif", writer="pillow", fps=10)

plt.show()
