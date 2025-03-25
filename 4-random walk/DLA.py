import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

N = 1000
L = 200
height = 500
launch_height = 0
kill_height = 150

grid = np.zeros((height, L), dtype=int)
grid[1, :] = 1


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_neighbors(grid, i, j):
    """Get the up, down, left and right neighbors of the cell"""
    neighbors = []

    # Up
    if i > 0:
        neighbors.append(grid[i - 1, j])

    # Down
    if i < grid.shape[0] - 1:
        neighbors.append(grid[i + 1, j])

    # Left
    if j > 0:
        neighbors.append(grid[i, j - 1])

    # Right
    if j < grid.shape[1] - 1:
        neighbors.append(grid[i, j + 1])

    return neighbors


def add_particle(color_value):
    hit = False

    while not hit:
        x = np.random.randint(0, L)
        y = np.random.randint(launch_height, kill_height)

        step_vectors = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
        ]

        while True:
            dx, dy = random.choice(step_vectors)

            x += dx
            y += dy

            # Wrap boundaries
            if x >= L:
                x = 0
            elif x < 0:
                x = L - 1

            # Check if the particle has hit the kill zone
            if y >= kill_height:
                break

            # Check if it has hit a neighbor
            neighbors = get_neighbors(grid, y, x)
            if len(neighbors) > 0:
                hit = True
                grid[y, x] = color_value
                break



color_range = np.linspace(0, 1, N)
for i in range(N):
    add_particle(i + 1)
    
grid = np.flipud(grid)

rainbow_cmap = plt.cm.get_cmap('rainbow_r')
colors = np.zeros((N + 1, 4))
colors[0] = [1, 1, 1, 1]
colors[1:] = rainbow_cmap(color_range)
custom_cmap = ListedColormap(colors)

plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap=custom_cmap, interpolation='none')
plt.title(f"2D Diffusion-Limited Aggregation (Particles: {N})")
plt.colorbar(label="Particle ID")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
