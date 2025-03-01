import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


time = 50000
L = 200

# Create an array of 200 zeros for height tracking
surface = np.zeros(L)
# Create a 3D array to track particle colors (0: no particle, 1: blue, 2: light blue)
particle_colors = np.zeros((1, L), dtype=int)


# Function to add a particle and track its color
def add_particle(position, color_value):
    global particle_colors
    height = int(surface[position])

    # Expand particle_colors array if needed
    if height >= particle_colors.shape[0]:
        additional_height = particle_colors.shape[0]  # Double the size
        zeros_to_append = np.zeros((additional_height, L), dtype=int)
        particle_colors = np.vstack((particle_colors, zeros_to_append))

    # Add the particle with its color
    particle_colors[height, position] = color_value
    surface[position] += 1


# Deposit particles with alternating colors
for i in range(time):
    # Choose a random integer between 0 and 199
    random_position = np.random.randint(0, L)

    # Determine color based on deposition time
    if (i // (time / 4)) % 2 == 0:
        color = 1  # Blue
    else:
        color = 2  # Light blue

    # Add the particle
    add_particle(random_position, color)

# Create a visualization
max_height = int(np.max(surface))

# Create a custom colormap: 0=white, 1=blue, 2=light blue
colors = ["white", "blue", "skyblue"]
cmap = ListedColormap(colors)

# Plot the pixel-based surface
plt.figure(figsize=(10, 6))

# Trim the particle_colors array to only include the heights we need
particle_colors_trimmed = particle_colors[:max_height, :]

# Then flip it so that height 0 is at the bottom
# particle_colors_reversed = np.flip(particle_colors_trimmed, axis=0)

plt.imshow(particle_colors_trimmed, cmap=cmap, interpolation="none", origin="lower")
plt.xlabel("Position")
plt.ylabel("Height")

# Set y-ticks to show actual heights (no need for custom calculation now)
num_ticks = 5  # Adjust this for more or fewer ticks
plt.yticks(np.linspace(0, max_height - 1, num_ticks).astype(int))

plt.title("Random Deposition Model")
plt.show()
