import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

N = 10000
L = 500
height = 500
launch_radius = 10
kill_radius = 20
spawn_zone = kill_radius - launch_radius
max_r = 1

grid = np.zeros((height, L), dtype=int)
grid[height // 2, L // 2] = 1


def get_neighbors(grid, i, j):
    """Get the up, down, left and right neighbors of the cell"""
    neighbors = []

    # Up
    if i > 0 and grid[i - 1, j] != 0:
        neighbors.append(grid[i - 1, j])

    # Down
    if i < grid.shape[0] - 1 and grid[i + 1, j] != 0:
        neighbors.append(grid[i + 1, j])

    # Left
    if j > 0 and grid[i, j - 1] != 0:
        neighbors.append(grid[i, j - 1])

    # Wrap left boundary
    elif j == 0 and grid[i, grid.shape[1] - 1] != 0:
        neighbors.append(grid[i, grid.shape[1] - 1])

    # Right
    if j < grid.shape[1] - 1 and grid[i, j + 1] != 0:
        neighbors.append(grid[i, j + 1])

    # Wrap right boundary
    elif j == grid.shape[1] - 1 and grid[i, 0] != 0:
        neighbors.append(grid[i, 0])

    return neighbors


def add_particle(color_value):
    global kill_radius, launch_radius, max_r
    hit = False

    while not hit:
        theta = np.random.uniform(0, 2 * np.pi)
        r0 = np.random.randint(launch_radius, kill_radius)

        center_x = L // 2
        center_y = height // 2

        x = int(center_x + r0 * np.cos(theta))
        y = int(center_y + r0 * np.sin(theta))

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

            # Check if the particle has hit the kill zone
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if r >= kill_radius:
                break

            # Check if it has hit a neighbor
            neighbors = get_neighbors(grid, y, x)
            if len(neighbors) > 0:
                hit = True
                grid[y, x] = color_value

                if r > max_r:
                    delta_r = r - max_r
                    max_r = r
                    kill_radius = min(height // 2, L // 2, kill_radius + delta_r)
                    launch_radius = min(
                        height // 2 - spawn_zone,
                        L // 2 - spawn_zone,
                        launch_radius + delta_r,
                    )
                break


color_range = np.linspace(0, 1, N)
for i in range(N):
    add_particle(i + 1)
    if (i + 1) % 500 == 0:
        print(f"Added {i + 1} particles")

grid = np.flipud(grid)

rainbow_cmap = plt.cm.get_cmap("rainbow_r")
colors = np.zeros((N + 1, 4))
colors[0] = [1, 1, 1, 1]
colors[1:] = rainbow_cmap(color_range)
custom_cmap = ListedColormap(colors)

plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap=custom_cmap, interpolation="none")
plt.title(f"2D Diffusion-Limited Aggregation (Particles: {N})")
plt.colorbar(label="Particle ID")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
