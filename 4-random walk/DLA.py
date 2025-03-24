import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

N = 1000
L = 200
height = 500
launch_radius = 100
kill_radius = 150

grid = np.zeros((height, L), dtype=int)
grid[-1, :] = 1


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def add_particle(color_value):
 pass


colors = ["darkgrey", "blue", "skyblue"]
cmap = ListedColormap(colors)
plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap=cmap, interpolation="none")
plt.title(f"2D Diffusion-Limited Aggregation (Particles: {N})")
plt.colorbar(ticks=[0, 1, 2], label="Particle Type")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
