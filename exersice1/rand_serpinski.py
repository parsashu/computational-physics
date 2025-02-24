import numpy as np
import matplotlib.pyplot as plt


# Number of points
n = 100000

# Create triangle points
vertices = np.array(
    [
        [0.0, 0.0],  # bottom left
        [0.5, 0.866],  # top point (using sin(60°) = 0.866)
        [1.0, 0.0],  # bottom right
    ]
)

A = vertices[0]
B = vertices[1]
C = vertices[2]

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


def random_point_generator():
    r1 = np.random.rand()
    r2 = np.random.rand()

    # Reflect (r1, r2) inside the triangle if r1 + r2 > 1
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2

    # (1-r1-r2)*A + r1*B + r2*C
    point = (1 - r1 - r2) * A + r1 * B + r2 * C
    return point


def serpinski_generator(iterations):
    global all_points
    
    for _ in range(iterations):
        point = random_point_generator()
        
        i = np.random.choice([0, 1, 2])
        chosen_vertex = vertices[i]
        
        new_point = scale_shape(point, 0.5, chosen_vertex)
        all_points = np.vstack((all_points, new_point))
    return all_points
    

    
    


all_points = serpinski_generator(n)

# Create the plot
plt.figure(figsize=(8, 6))

# Plot initial triangle points
plt.scatter(
    vertices[:, 0],
    vertices[:, 1],
    color="blue",
    s=100,
    label="Triangle vertices",
)

plt.scatter(all_points[:, 0], all_points[:, 1], color="red", s=1, label="Random point")

# Plot triangle edges
plt.plot(
    [vertices[0, 0], vertices[1, 0]],
    [vertices[0, 1], vertices[1, 1]],
    "b-",
)
plt.plot(
    [vertices[1, 0], vertices[2, 0]],
    [vertices[1, 1], vertices[2, 1]],
    "b-",
)
plt.plot(
    [vertices[2, 0], vertices[0, 0]],
    [vertices[2, 1], vertices[0, 1]],
    "b-",
)

plt.legend()
plt.axis("equal")
plt.grid(True)
plt.title("Triangle with Random Point")
plt.show()
