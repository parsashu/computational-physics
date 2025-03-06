import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


N = 1000
L = 200
max_height = 300

surface = np.zeros(L)
particle_colors = np.zeros((max_height, L), dtype=int)


def add_particle(position, color_value):
    """Function to add a particle and track its color"""
    global particle_colors
    height = 2 * max_height

    while True:
        bar_height = int(surface[position])
        # Check if it's a hit
        if bar_height >= height:
            particle_colors[bar_height, position] = color_value
            surface[position] += 1
            break

        # If it's not a hit, move the particle
        if position == 0:
            position = L - 1
        else:
            position -= 1
        height -= 1


for i in range(N):
    if max(surface) >= max_height:
        break
    random_position = np.random.randint(0, L)

    if (i // (N / 4)) % 2 == 0:
        color = 1  # Blue
    else:
        color = 2  # Light blue

    add_particle(random_position, color)


colors = ["white", "blue", "skyblue"]
cmap = ListedColormap(colors)

plt.figure(figsize=(10, 6))

particle_colors_trimmed = particle_colors[:max_height, :]

plt.imshow(particle_colors_trimmed, cmap=cmap, interpolation="none", origin="lower")
plt.xlabel("Position")
plt.ylabel("Height")

num_ticks = 5
plt.yticks(np.linspace(0, max_height - 1, num_ticks).astype(int))

plt.title(f"Competitive Deposition Model (Particles: {N})")
plt.show()

# Create a histogram of surface heights
plt.figure(figsize=(10, 6))
plt.hist(surface, bins=30, color="blue", alpha=0.7, edgecolor="black")
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.title("Histogram of Surface Heights")
plt.grid(True, alpha=0.3)
plt.show()
