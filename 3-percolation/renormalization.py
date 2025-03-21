import numpy as np
import matplotlib.pyplot as plt

L = 50
p = 0.5

grid = np.zeros((L, L))


def turn_on(grid, p):
    """Open sites with probability p"""
    random_values = np.random.random((L, L))
    grid = np.where(random_values < p, 1, 0)
    return grid


def renormalize(grid):
    """
    Renormalize the grid by applying the rule:
    - If there's a path connecting top to bottom in a 2x2 block, the resulting cell is filled (1)
    - Otherwise, the resulting cell is empty (0)
    """
    L_old = grid.shape[0]
    L_new = L_old // 2
    new_grid = np.zeros((L_new, L_new))

    for i in range(L_new):
        for j in range(L_new):
            block = grid[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]

            top_filled = np.any(block[0, :])
            bottom_filled = np.any(block[1, :])
            col0_filled = block[0, 0] and block[1, 0]
            col1_filled = block[0, 1] and block[1, 1]

            if (top_filled and bottom_filled) or col0_filled or col1_filled:
                new_grid[i, j] = 1

    return new_grid


grid_on = turn_on(grid, p)
renormalized_grid = renormalize(grid_on)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
colors_with_alpha = [(0.3, 0.3, 0.3, 1.0), (0, 0, 0.5, 1.0)]
cmap = plt.matplotlib.colors.ListedColormap(colors_with_alpha)
plt.imshow(grid_on, cmap=cmap, vmin=0, vmax=1)
plt.clim(-0.5, 1.5)
plt.grid(False)
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.1)
plt.xticks([])
plt.yticks([])
plt.title(f"Original Grid (L={L}, p={p})")

plt.subplot(1, 2, 2)
plt.imshow(renormalized_grid, cmap=cmap, vmin=0, vmax=1)
plt.clim(-0.5, 1.5)
plt.grid(False)
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, L // 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, L // 2, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.1)
plt.xticks([])
plt.yticks([])
plt.title(f"Renormalized Grid (L={L//2})")

plt.tight_layout()
plt.show()
