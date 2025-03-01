import numpy as np
import matplotlib.pyplot as plt

time = 8000

# Create an array of 200 zeros
surface = np.zeros(200)


for i in range(time):
    # Choose a random integer between 0 and 199
    random_position = np.random.randint(0, 200)

    # Set the value of the random position to 1
    surface[random_position] += 1

# Create a pixel-based visualization
max_height = int(np.max(surface))
pixel_grid = np.zeros((max_height, 200))

# Fill the pixel grid based on surface heights
for x in range(200):
    height = int(surface[x])
    for y in range(height):
        pixel_grid[max_height - 1 - y, x] = 1

# Plot the pixel-based surface
plt.figure(figsize=(10, 6))
plt.imshow(pixel_grid, cmap="Blues", interpolation="none")
plt.xlabel("Position")
plt.ylabel("Height")
plt.title("Random Deposition Model")
plt.savefig("random_deposition_pixels.png")
plt.show()
