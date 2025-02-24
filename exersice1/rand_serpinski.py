import numpy as np
import matplotlib.pyplot as plt

np.random.seed(20)

iterations = 7

# Create triangle points
initial_points = np.array(
    [
        [0.0, 0.0],  # bottom left
        [0.5, 0.866],  # top point (using sin(60°) = 0.866)
        [1.0, 0.0],  # bottom right
    ]
)

A = initial_points[0]
B = initial_points[1]
C = initial_points[2]

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


def random_point_generator():
    r1 = np.random.rand()
    r2 = np.random.rand()

    # Reflect (r1, r2) inside the triangle if r1 + r2 > 1
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2

    # (1-r1-r2)*A + r1*B + r2*C
    point = (1 - r1 - r2) * A + r1 * B + r2 * C
    return point


# Generate a random point
point = random_point_generator()

# Create the plot
plt.figure(figsize=(8, 6))

# Plot initial triangle points
plt.scatter(
    initial_points[:, 0],
    initial_points[:, 1],
    color="blue",
    s=100,
    label="Triangle vertices",
)

# Plot random point
plt.scatter(point[0], point[1], color="red", s=100, label="Random point")

# Plot triangle edges
plt.plot(
    [initial_points[0, 0], initial_points[1, 0]],
    [initial_points[0, 1], initial_points[1, 1]],
    "b-",
)
plt.plot(
    [initial_points[1, 0], initial_points[2, 0]],
    [initial_points[1, 1], initial_points[2, 1]],
    "b-",
)
plt.plot(
    [initial_points[2, 0], initial_points[0, 0]],
    [initial_points[2, 1], initial_points[0, 1]],
    "b-",
)

plt.legend()
plt.axis("equal")
plt.grid(True)
plt.title("Triangle with Random Point")
plt.show()
