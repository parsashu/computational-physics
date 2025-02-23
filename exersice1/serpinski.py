import numpy as np
import matplotlib.pyplot as plt

iterations = 1

# Create triangle points
initial_points = np.array(
    [
        [0.0, 0.0],  # bottom left
        [0.5, 0.866],  # top point (using sin(60°) = 0.866)
        [1.0, 0.0],  # bottom right
    ]
)

all_points = initial_points.copy()


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


def serpinski(all_points, iterations):
    global initial_points

    for _ in range(iterations):
        left_points = scale_shape(all_points.copy(), 0.5, initial_points[0])
        top_points = scale_shape(all_points.copy(), 0.5, initial_points[1])
        right_points = scale_shape(all_points.copy(), 0.5, initial_points[2])
        all_points = np.vstack([left_points, right_points, top_points])
        # unique_points = np.unique(all_points, axis=0)

    return left_points, top_points, right_points, all_points


left_points, top_points, right_points, all_points = serpinski(all_points, iterations)


# Create figure with white background
plt.figure(figsize=(8, 6), facecolor="white")

for i in range(len(all_points) // 3):
    triangle = all_points[3 * i : 3 * i + 3]
    plt.fill(triangle[:, 0], triangle[:, 1], alpha=0.3, color="blue", label="Top")

plt.scatter(all_points[:, 0], all_points[:, 1], c="blue", marker="o", s=100)


# Plot all points in black with numbers
plt.scatter(all_points[:, 0], all_points[:, 1], c="blue", marker="o", s=100)

# Add point numbers for all points
for i, (x, y) in enumerate(all_points):
    plt.annotate(f"{i}", (x, y), xytext=(5, 5), textcoords="offset points")


plt.axis("equal")
plt.axis("on")
plt.grid(False)

plt.show()
