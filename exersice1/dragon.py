import numpy as np
import matplotlib.pyplot as plt


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
    initial_points = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
        ]
    )
    # Rotate and scale initial points (+45 degrees)
    rotated_points1 = rotate_shape(initial_points, 45, [0.0, 0.0])
    scaled_points1 = scale_shape(rotated_points1, 0.5, [0.0, 0.0])
    
    # Rotate and scale initial points (-45 degrees)
    rotated_points2 = rotate_shape(initial_points, 135, [0.0, 0.0])
    scaled_points2 = scale_shape(rotated_points2, 0.5, [0.0, 0.0])
    
    # Store initial blue points
    blue_points = scaled_points1.copy()
    red_points = scaled_points2.copy()

    return np.array(blue_points), np.array(red_points)


# Create figure with white background
plt.figure(figsize=(8, 4), facecolor="white")

# Plot one iteration
blue_points, red_points = dragon_curve(1)

plt.plot(blue_points[:, 0], blue_points[:, 1], "b-")  # Blue line
plt.plot(red_points[:, 0], red_points[:, 1], "r-")  # Red line

plt.axis("equal")
plt.axis("on")

plt.show()
