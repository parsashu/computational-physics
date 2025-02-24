import numpy as np
import matplotlib.pyplot as plt

iterations = 16
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