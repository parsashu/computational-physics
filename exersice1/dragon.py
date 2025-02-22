import numpy as np
import matplotlib.pyplot as plt


def rotate_point(point, angle_degrees):
    """Rotate a point by given angle in degrees"""
    theta = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return np.dot(rotation_matrix, point)


def dragon_curve(iterations):
    # Start with a single segment
    points = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]

    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]

            # Midpoint
            mid = (start + end) / 2

            # Vector from start to end
            vector = end - start

            # Rotate the second half 90 degrees clockwise around midpoint
            rotated = rotate_point(vector / 2, -90)

            new_points.extend([start, mid + rotated])
        new_points.append(points[-1])
        points = new_points

    return np.array(points)


# Create figure with white background
plt.figure(figsize=(8, 4), facecolor="white")

# Plot one iteration
points = dragon_curve(1)
plt.plot(points[:, 0], points[:, 1], "b-")  # Blue line
plt.plot(points[:, 0], points[:, 1], "ro")  # Red points

plt.axis("equal")
plt.axis("off")

plt.show()
