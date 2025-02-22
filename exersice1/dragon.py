import numpy as np
import matplotlib.pyplot as plt

iterations = 1
initial_points = np.array(
    [
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 0.0],
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


def dragon_curve(iterations):
    points = initial_points.copy()
    blue_points = points.copy()[:2]
    red_points = points.copy()[1:]

    for _ in range(iterations - 1):
        last_point = points[-1]

        # Rotate and scale initial points (45 degrees)
        rotated_points1 = rotate_shape(points, 45, last_point)
        blue_points = scale_shape(rotated_points1, 0.5, last_point)

        # Rotate and scale initial points (135 degrees)
        rotated_points2 = rotate_shape(points, 135, last_point)
        red_points = scale_shape(rotated_points2, 0.5, last_point)

        # Store initial blue points
        reversed_red_points = red_points[::-1]
        points = np.vstack((blue_points, reversed_red_points))

    return blue_points, red_points


# Create figure with white background
plt.figure(figsize=(8, 4), facecolor="white")

# Plot one iteration
blue_points, red_points = dragon_curve(iterations)


plt.plot(blue_points[:, 0], blue_points[:, 1], "b-")  # Blue line
plt.plot(red_points[:, 0], red_points[:, 1], "r-")  # Red line

plt.axis("equal")
plt.axis("off")

plt.show()
