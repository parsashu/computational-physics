import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random


N = 5000
L = 1000
height = 1000

launch_l = 5
kill_l = 10
spawn_zone_l = kill_l - launch_l

max_x = 1
max_y = 1

grid = np.zeros((height, L), dtype=int)
center_x = L // 2
center_y = height // 2
grid[center_y, center_x] = 1


def has_neighbor(x, y):
    positions = [
        ((y - 1) % height, x),  # Up
        ((y + 1) % height, x),  # Down
        (y, (x - 1) % L),  # Left
        (y, (x + 1) % L),  # Right
    ]

    for ny, nx in positions:
        if grid[ny, nx] != 0:
            return True
    return False


def add_particle(color_value):
    global kill_l, launch_l, max_x, max_y
    hit = False

    while not hit:
        if np.random.random() < 0.5:
            x = np.random.randint(center_x + launch_l, center_x + kill_l)
        else:
            x = np.random.randint(center_x - kill_l, center_x - launch_l)

        if np.random.random() < 0.5:
            y = np.random.randint(center_y + launch_l, center_y + kill_l)
        else:
            y = np.random.randint(center_y - kill_l, center_y - launch_l)

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
            if dist_x >= kill_l or dist_y >= kill_l:
                break

            # Check if it has hit a neighbor
            if has_neighbor(x, y):
                hit = True
                grid[y, x] = color_value

                if max(dist_x, dist_y) > max(max_x, max_y):
                    max_x = max(max_x, dist_x)
                    max_y = max(max_y, dist_y)
                    kill_l = min(height // 2, L // 2, kill_l + 1)
                    launch_l = min(
                        height // 2 - spawn_zone_l,
                        L // 2 - spawn_zone_l,
                        launch_l + 1,
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
plt.title(f"2D Diffusion-Limited Aggregation quadratic (Particles: {N})")
plt.colorbar(label="Particle ID")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
