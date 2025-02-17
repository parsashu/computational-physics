import numpy as np
import matplotlib.pyplot as plt

# Define three endpoints
start_point = np.array([-0.5, 0])
end_point = np.array([0.5, 0])


def find_devider_points(start_point, end_point):
    direction = end_point - start_point
    line_length = np.linalg.norm(end_point - start_point)
    unit_vector = direction / line_length

    point1 = start_point + (line_length / 3) * unit_vector
    point2 = start_point + (2 * line_length / 3) * unit_vector
    return point1, point2


point1, point2 = find_devider_points(start_point, end_point)


def find_middle_point(start_point, end_point):
    direction = end_point - start_point
    line_length = np.linalg.norm(end_point - start_point)
    unit_vector = direction / line_length
    return start_point + (line_length / 2) * unit_vector


middle_point = find_middle_point(start_point, end_point)


def find_top_point(start_point, end_point):
    direction = start_point - end_point
    line_length = np.linalg.norm(start_point - end_point)

    unit_vector = direction / line_length
    perpendicular = np.array([unit_vector[1], -unit_vector[0]])

    middle_point = find_middle_point(start_point, end_point)
    triangle_height = line_length * np.sqrt(3) / 6
    return middle_point + triangle_height * perpendicular


top_point = find_top_point(start_point, end_point)


# Generate points between the endpoints for first line
num_points = 100  # Number of points to create smooth lines
t = np.linspace(0, 1, num_points)[:, np.newaxis]

# Parametric equations for first line
line1 = start_point + (point1 - start_point) * t
line2 = point1 + (top_point - point1) * t
line3 = top_point + (point2 - top_point) * t
line4 = point2 + (end_point - point2) * t



# Create the plot
plt.figure(figsize=(8, 4))

# Plot both lines
plt.plot(line1[:, 0], line1[:, 1], "b-")  # Plot x and y coordinates separately
plt.plot(line2[:, 0], line2[:, 1], "b-")
plt.plot(line3[:, 0], line3[:, 1], "b-")
plt.plot(line4[:, 0], line4[:, 1], "b-")

# Plot endpoints
plt.plot(
    [start_point[0], end_point[0]], [start_point[1], end_point[1]], "ro"
)  # Show endpoints as red dots
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], "ro")
# plt.plot([middle_point[0]], [middle_point[1]], "ro")
plt.plot([top_point[0]], [top_point[1]], "ro")

plt.axis("equal")
plt.grid(True)
plt.show()
