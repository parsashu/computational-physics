import numpy as np
import matplotlib.pyplot as plt

# Define three endpoints
x1, y1 = -5, 0  # First point
x2, y2 = 0, 0  # Second point
x3, y3 = 5, 5  # Second point
x4, y4 = 10, 0  # Third point
x5, y5 = 15, 0  # Fourth point

# Generate points between the endpoints for first line
num_points = 100  # Number of points to create smooth lines
t = np.linspace(0, 1, num_points)

# Parametric equations for first line
x_line1 = x1 + (x2 - x1) * t
y_line1 = y1 + (y2 - y1) * t

# Parametric equations for second line
x_line2 = x2 + (x3 - x2) * t
y_line2 = y2 + (y3 - y2) * t

# Parametric equations for third line
x_line3 = x3 + (x4 - x3) * t
y_line3 = y3 + (y4 - y3) * t

# Parametric equations for fourth line
x_line4 = x4 + (x5 - x4) * t
y_line4 = y4 + (y5 - y4) * t

# Create the plot
plt.figure(figsize=(8, 4))

# Plot both lines
plt.plot(x_line1, y_line1, "b-")
plt.plot(x_line2, y_line2, "b-")
plt.plot(x_line3, y_line3, "b-")
plt.plot(x_line4, y_line4, "b-")

# Plot endpoints
plt.plot([x1, x2, x3, x4, x5], [y1, y2, y3, y4, y5], "ro")  # Show endpoints as red dots

plt.axis("equal")
plt.grid(True)
plt.show()
