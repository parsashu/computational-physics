import numpy as np
import matplotlib.pyplot as plt


n_points = 10000  # Number of points
p = 20
right_angle = -45
left_angle = 45
top_angle = -10

# Create square points
vertices = np.array(
    [
        [0.0, 0.0],
        [0.0, 4.0],
        [2.0, 4.0],
        [2.0, 0.0],
    ]
)

A = vertices[0]
B = vertices[1]
C = vertices[2]
D = vertices[3]

all_points = vertices.copy()


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


def rotate_shape(points, angle_degrees, center_point=None):
    """Rotate points around a center point"""

    # If no center point given, use center of points
    if center_point is None:
        center_point = np.mean(points, axis=0)

    # Convert angle to radians
    theta = np.radians(angle_degrees)

    # Create rotation matrix
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Translate to origin
    translated = points - center_point

    # Rotate points
    rotated = np.dot(translated, rotation_matrix.T)

    # Translate back
    return rotated + center_point


def random_point_generator():
    # Generate random x and y coordinates within the square bounds
    x = np.random.uniform(0, 2.0)  # Random x between 0 and 2
    y = np.random.uniform(0, 4.0)  # Random y between 0 and 4
    return np.array([x, y])


new_points = rotate_shape(vertices, 0, A)
point = random_point_generator()

# Plot the square
plt.figure(figsize=(8, 6))
# Add the first point to the end to close the shape
vertices_closed = np.vstack([new_points, new_points[0]])
plt.plot(
    vertices_closed[:, 0], vertices_closed[:, 1], "k-"
)  # Draw lines connecting vertices
plt.fill(
    vertices_closed[:, 0], vertices_closed[:, 1], alpha=0.1
)  # Fill with transparent color

plt.scatter(point[0], point[1], s=1, color="red")

plt.grid(True)
plt.axis("equal")  # Make sure the aspect ratio is equal
plt.title("Square")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
