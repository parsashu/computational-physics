import numpy as np
import matplotlib.pyplot as plt

iterations = 1

# Create triangle points
all_points = np.array(
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
    global all_points
    
    left_points = scale_shape(all_points.copy(), 0.5, all_points[0])
    top_points = scale_shape(all_points.copy(), 0.5, all_points[1])
    right_points = scale_shape(all_points.copy(), 0.5, all_points[2])

    all_points = np.vstack([left_points, right_points, top_points])
    return left_points, top_points, right_points


left_points, top_points, right_points = serpinski(iterations)

# Create figure with white background
plt.figure(figsize=(8, 8), facecolor="white")

# Plot and fill each triangle separately
plt.fill(left_points[:, 0], left_points[:, 1], alpha=0.3, color='blue', label='Left')
plt.fill(right_points[:, 0], right_points[:, 1], alpha=0.3, color='red', label='Right')
plt.fill(top_points[:, 0], top_points[:, 1], alpha=0.3, color='green', label='Top')

# Plot points for each triangle
plt.scatter(left_points[:, 0], left_points[:, 1], c='blue', marker='o', s=100)
plt.scatter(right_points[:, 0], right_points[:, 1], c='red', marker='o', s=100)
plt.scatter(top_points[:, 0], top_points[:, 1], c='green', marker='o', s=100)

# Add point numbers for each set
for i, (x, y) in enumerate(left_points):
    plt.annotate(f'L{i}', (x, y), xytext=(5, 5), textcoords='offset points')
for i, (x, y) in enumerate(right_points):
    plt.annotate(f'R{i}', (x, y), xytext=(5, 5), textcoords='offset points')
for i, (x, y) in enumerate(top_points):
    plt.annotate(f'T{i}', (x, y), xytext=(5, 5), textcoords='offset points')


plt.axis("equal")
plt.axis("on")
plt.grid(False)

plt.show()
