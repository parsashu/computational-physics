import numpy as np
import matplotlib.pyplot as plt

L = 10

grid = np.zeros((L, L))


def turn_on(grid, p):
    """Open sites with probability p"""
    random_values = np.random.random((L, L))
    grid = np.where(random_values < p, 1, 0)
    return grid


gird_on = turn_on(grid, 0.5)


plt.figure(figsize=(6, 6))
colors = ["purple", "blue"]
colors_with_alpha = [(0.8, 0, 0.8, 0.7), (0, 0, 1, 0.7)]
cmap = plt.matplotlib.colors.ListedColormap(colors_with_alpha)
plt.imshow(gird_on, cmap=cmap, vmin=0, vmax=1)
plt.clim(-0.5, 1.5)
plt.grid(False)
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
plt.show()
