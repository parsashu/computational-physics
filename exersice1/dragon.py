import numpy as np
import matplotlib.pyplot as plt


def rotate_point(point, angle_degrees):
    """Rotate a point by given angle in degrees"""
    theta = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return np.dot(rotation_matrix, point)


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


def dragon_curve(iterations):
    # Start with a single segment
    points = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
        ]
    )

    return np.array(points)


# Create figure with white background
plt.figure(figsize=(8, 4), facecolor="white")

# Plot one iteration
points = dragon_curve(1)
scaled = scale_shape(points, 0.5, [0.0, 0.0])
plt.plot(scaled[:, 0], scaled[:, 1], "b-")  # Blue line
plt.plot(scaled[:, 0], scaled[:, 1], "ro")  # Red points

plt.axis("equal")
plt.axis("on")

plt.show()
