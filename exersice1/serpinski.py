import numpy as np
import matplotlib.pyplot as plt

iterations = 1

# Create triangle points
triangle_points = np.array(
    [
        [0.0, 0.0],  # bottom left
        [0.5, 0.866],  # top point (using sin(60°) = 0.866)
        [1.0, 0.0],  # bottom right
    ]
)


def scale_shape(points, scale_factor, center_point=None):
    """Scale points around a center point"""

    # If no center point given, use center of points
    if center_point is None:
        center_point = np.mean(points, axis=0)

    # Translate to origin
    translated = points - center_point

    # Create scaling matrix
    scaling_matrix = np.array([[scale_factor, 0], [0, scale_factor]])

    # Scale points
    scaled = np.dot(translated, scaling_matrix)

    # Translate back
    return scaled + center_point


def serpinski(iterations):
    triangle_points = scale_shape(triangle_points.copy(), 0.5, [0, 0])


# Create figure with white background
plt.figure(figsize=(8, 8), facecolor="white")

# Fill the triangle
plt.fill(triangle_points[:, 0], triangle_points[:, 1], alpha=0.3, color="blue")

# Plot points
plt.scatter(triangle_points[:, 0], triangle_points[:, 1], c="blue", marker="o", s=100)

# Add point numbers
for i, (x, y) in enumerate(triangle_points):
    plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points")

plt.axis("equal")
plt.axis("on")
plt.grid(False)

plt.show()
