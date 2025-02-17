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


def line_between_points(start_point, end_point):
    num_points = 100  # Number of points to create smooth lines
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    return start_point + (end_point - start_point) * t


line1 = line_between_points(start_point, point1)
line2 = line_between_points(point1, top_point)
line3 = line_between_points(top_point, point2)
line4 = line_between_points(point2, end_point)


# Create the plot
plt.figure(figsize=(8, 4))


# Plot lines
def plot_line(line):
    plt.plot(line[:, 0], line[:, 1], "b-")


plot_line(line1)
plot_line(line2)
plot_line(line3)
plot_line(line4)


# Plot points
def plot_point(point):
    plt.plot(point[0], point[1], "ro")


plot_point(start_point)
plot_point(end_point)
plot_point(point1)
plot_point(point2)
plot_point(top_point)


plt.axis("equal")
plt.grid(True)
plt.show()
