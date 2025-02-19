import numpy as np
import matplotlib.pyplot as plt

iterations = 5
start_point = np.array([-0.5, 0])
end_point = np.array([0.5, 0])
all_points = [start_point, end_point]
temp_points = all_points.copy()
all_lines = []


# Point finder functions
def find_devider_points(start_point, end_point):
    direction = end_point - start_point
    line_length = np.linalg.norm(end_point - start_point)
    unit_vector = direction / line_length

    point1 = start_point + (line_length / 3) * unit_vector
    point2 = start_point + (2 * line_length / 3) * unit_vector
    return point1, point2


def find_middle_point(start_point, end_point):
    direction = end_point - start_point
    line_length = np.linalg.norm(end_point - start_point)
    unit_vector = direction / line_length
    return start_point + (line_length / 2) * unit_vector


def find_top_point(start_point, end_point):
    direction = start_point - end_point
    line_length = np.linalg.norm(start_point - end_point)

    unit_vector = direction / line_length
    perpendicular = np.array([unit_vector[1], -unit_vector[0]])

    middle_point = find_middle_point(start_point, end_point)
    triangle_height = line_length * np.sqrt(3) / 6
    return middle_point + triangle_height * perpendicular


def find_new_points(start_point, end_point):
    point1, point2 = find_devider_points(start_point, end_point)
    top_point = find_top_point(start_point, end_point)
    return point1, point2, top_point


def add_new_points(start_point, end_point):
    global temp_points
    point1, point2, top_point = find_new_points(start_point, end_point)

    # Find the index of end_point
    for i, point in enumerate(temp_points):
        if np.array_equal(point, end_point):
            # Insert new points at position i
            temp_points.insert(i, point1)
            temp_points.insert(i + 1, top_point)
            temp_points.insert(i + 2, point2)
            break


def update_all_points():
    global all_points
    for i in range(len(all_points) - 1):
        add_new_points(all_points[i], all_points[i + 1])

    all_points = temp_points.copy()


# Line functions
def line_between_points(start_point, end_point):
    num_points = 100  # Number of points to create smooth lines
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    return start_point + (end_point - start_point) * t


def add_new_lines():
    for i in range(len(all_points) - 1):
        line = line_between_points(all_points[i], all_points[i + 1])
        all_lines.append(line)


# Create the plot
plt.figure(figsize=(8, 4))


# Plot functions
def plot_lines():
    for line in all_lines:
        plt.plot(line[:, 0], line[:, 1], "b-")


# Plot points
def plot_points():
    for point in all_points:
        plt.plot(point[0], point[1], "ro")


def koch(iterations, points=False, lines=True):
    for i in range(iterations):
        update_all_points()        
    add_new_lines()
    
    if points:
        plot_points()
    if lines:
        plot_lines()


koch(iterations)

#test



plt.axis("equal")
plt.axis("off")

