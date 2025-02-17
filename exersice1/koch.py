import numpy as np
import matplotlib.pyplot as plt

# Define three endpoints
start_point = np.array([-0.5, 0])
end_point = np.array([0.5, 0])


# Generate points between the endpoints for first line
num_points = 100  # Number of points to create smooth lines
t = np.linspace(0, 1, num_points)[:, np.newaxis]

# Parametric equations for first line
line1 = start_point + (end_point - start_point) * t


def find_on_line_dots(start_point, end_point):
    pass


# Create the plot
plt.figure(figsize=(8, 4))

# Plot both lines
plt.plot(line1[:, 0], line1[:, 1], "b-")  # Plot x and y coordinates separately

# Plot endpoints
plt.plot(
    [start_point[0], end_point[0]], [start_point[1], end_point[1]], "ro"
)  # Show endpoints as red dots


plt.axis("equal")
plt.grid(True)
plt.show()
