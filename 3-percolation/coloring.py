import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

L = 50
p = 0.5
int_max = 1000
color = 2

grid = np.zeros((L, L))
grid[:, 0] = 1
grid[:, -1] = int_max


def turn_on(grid, p):
    """Open sites with probability p"""
    random_values = np.random.random((L, L))
    grid = np.where(random_values < p, 1, 0)
    return grid



def get_neighbors(grid, i, j):
    neighbors = {}
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue  # Skip the current cell
            ni, nj = i + di, j + dj
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                neighbors[(ni, nj)] = grid[ni, nj]
    return neighbors



random_values = np.random.random((L, L))

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if random_values[i, j] < p:
            grid[i, j] = color
            color += 1
            
            neighbors = get_neighbors(grid, i, j)
            # print(neighbors)

a = get_neighbors(grid, 0, 0)
print(a)


plt.figure(figsize=(6, 6))
base_cmap = cm.get_cmap("tab20", 101)
colors = [base_cmap(i) for i in range(101)]
colors[0] = (0.3, 0.3, 0.3, 1.0)
colors[1] = (0.3, 0.3, 0.3, 1.0)
cmap = ListedColormap(colors)
plt.imshow(grid, cmap=cmap, vmin=0, vmax=int_max)
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
