import numpy as np
import matplotlib.pyplot as plt

# Define two endpoints
x1, y1 = 0, 0  # First point
x2, y2 = 5, 0  # Second point

# Generate points between the endpoints
num_points = 100  # Number of points to create a smooth line
t = np.linspace(0, 1, num_points)

# Parametric equations for line
x = x1 + (x2 - x1) * t
y = y1 + (y2 - y1) * t

# Create the plot
plt.figure(figsize=(8, 8))
plt.plot(x, y, 'b-')

# Plot endpoints
plt.plot([x1, x2], [y1, y2], 'ro')  # Show endpoints as red dots

plt.axis('off')
plt.show()