import numpy as np
import matplotlib.pyplot as plt


n_points = 100000  # Number of points
p = 20
right_angle = -50
left_angle = 50
top_angle = -3

# Create square points
vertices = np.array(
    [
        [0.0, 0.0],
        [0.0, 4.0],
        [2.0, 4.0],
        [2.0, 0.0],
        [0.0, 0.0],
    ]
)

all_points = vertices.copy()


def rotate_shape(points, angle_degrees, center_point=None):
    """Rotate points around a center point"""

    # If no center point given, use center of points
    if center_point is None:
        center_point = np.mean(points, axis=0)

    # Convert angle to radians
    theta = np.radians(angle_degrees)

    # Create rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    # Translate to origin
    translated = points - center_point

    # Rotate points
    rotated = np.dot(translated, rotation_matrix.T)

    # Translate back
    return rotated + center_point


def scale_shape(points, scale, center_point=None, tale=False):
    # If no center point given, use center of points
    if center_point is None:
        center_point = np.mean(points, axis=0)

    # Translate to origin
    translated = points - center_point

    # If tale, make width 0
    if tale:
        scaling_matrix = np.array([[0, 0], [0, scale]])
    else:
        scaling_matrix = np.array([[scale, 0], [0, scale]])

    scaled = np.dot(translated, scaling_matrix)

    # Translate back
    points[:] = scaled + center_point
    return points


def random_point_generator():
    # Generate random x and y coordinates within the square bounds
    x = np.random.uniform(0, 2.0)  # Random x between 0 and 2
    y = np.random.uniform(0, 4.0)  # Random y between 0 and 4
    return np.array([x, y])


# Main functions
def right_func(points):
    points_copy = points.copy()
    rotated = rotate_shape(points_copy, right_angle)
    scaled = scale_shape(rotated, 0.35, [1.5, 0.39])
    return scaled


def left_func(points):
    points_copy = points.copy()
    rotated = rotate_shape(points_copy, left_angle)
    scaled = scale_shape(rotated, 0.4, [0.7, 0.80])
    return scaled


def top_func(points):
    points_copy = points.copy()
    rotated = rotate_shape(points_copy, top_angle)
    scaled = scale_shape(rotated, 0.82, [1.4, 4.3])
    return scaled


def tail_func(points):
    points_copy = points.copy()
    rotated = rotate_shape(points_copy, top_angle)
    scaled = scale_shape(rotated, 0.19, [1.17, 0], tale=True)
    return scaled


def sarakhs(n_points, p):
    global all_points

    for _ in range(n_points):
        point = random_point_generator()

        for _ in range(p):
            random_func = np.random.choice(
                [right_func, left_func, top_func, tail_func], p=[0.32, 0.32, 0.32, 0.04]
            )
            new_point = random_func(point)
            point = new_point

        all_points = np.vstack((all_points, point))
    return all_points


right = right_func(vertices)
left = left_func(vertices)
top = top_func(vertices)
tail = tail_func(vertices)

all_points = sarakhs(n_points, p)

# Plot the square
plt.figure(figsize=(8, 6))


# Plot all generated points
plt.scatter(all_points[:, 0], all_points[:, 1], c="green", s=1, alpha=1)

plt.grid(False)
plt.axis("equal")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
