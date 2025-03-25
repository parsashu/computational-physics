import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

N = 10000
L = 1000
height = 1000

launch_x = 10
launch_y = 10
kill_x = 30
kill_y = 30

spawn_zone_x = kill_x - launch_x
spawn_zone_y = kill_y - launch_y

max_x = 1
max_y = 1

grid = np.zeros((height, L), dtype=int)
center_x = L // 2
center_y = height // 2
grid[center_y, center_x] = 1


def get_neighbors(grid, i, j):
    """Get the up, down, left and right neighbors of the cell"""
    neighbors = []

    # Up
    if i > 0 and grid[i - 1, j] != 0:
        neighbors.append(grid[i - 1, j])
    # Wrap up boundary
    elif i == 0 and grid[grid.shape[0] - 1, j] != 0:
        neighbors.append(grid[grid.shape[0] - 1, j])

    # Down
    if i < grid.shape[0] - 1 and grid[i + 1, j] != 0:
        neighbors.append(grid[i + 1, j])
    # Wrap down boundary
    elif i == grid.shape[0] - 1 and grid[0, j] != 0:
        neighbors.append(grid[0, j])

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
    global kill_x, kill_y, launch_x, launch_y, max_x, max_y
    hit = False

    while not hit:
        if np.random.random() < 0.5:
            x = np.random.randint(center_x + launch_x, center_x + kill_x)
        else:
            x = np.random.randint(center_x - kill_x, center_x - launch_x)

        if np.random.random() < 0.5:
            y = np.random.randint(center_y + launch_y, center_y + kill_y)
        else:
            y = np.random.randint(center_y - kill_y, center_y - launch_y)

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
            x %= L
            y %= height

            dist_x = abs(x - center_x)
            dist_y = abs(y - center_y)

            # Check if the particle has hit the kill zone
            if dist_x >= kill_x or dist_y >= kill_y:
                break

            # Check if it has hit a neighbor
            neighbors = get_neighbors(grid, y, x)
            if len(neighbors) > 0:
                hit = True
                grid[y, x] = color_value

                if dist_x > max_x:
                    max_x = dist_x
                    kill_x = min(height // 2, L // 2, kill_x + 1)
                    launch_x = min(
                        height // 2 - spawn_zone_x,
                        L // 2 - spawn_zone_x,
                        launch_x + 1,
                    )
                elif dist_y > max_y:
                    max_y = dist_y
                    kill_y = min(height // 2, L // 2, kill_y + 1)
                    launch_y = min(
                        height // 2 - spawn_zone_y,
                        L // 2 - spawn_zone_y,
                        launch_y + 1,
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
