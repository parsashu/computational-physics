import matplotlib.pyplot as plt
import numpy as np


# Specify the number of rows for Khayyam's triangle
n = 4


def generate_khayyam_triangle(n):
    triangle = []
    for i in range(n):
        # Create a new row with i+1 elements, all initially set to 1.
        row = [1] * (i + 1)
        # Calculate the inner numbers (if the row has at least 3 elements).
        for j in range(1, i):
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
        triangle.append(row)
    return triangle


def generate_point(triangle):
    # Get the dimensions of the triangle
    height = len(triangle)
    width = 2 * height - 1  # Width needed for centered triangle
    
    # Create a matrix to store the colors (1 for odd, 0 for even)
    colors = np.zeros((height, width))
    
    # Fill the matrix with 1s (odd) and 0s (even) in a triangular pattern
    for i in range(height):
        # Calculate starting position for this row to center it
        start_pos = (width - (2 * i + 1)) // 2
        for j in range(len(triangle[i])):
            colors[i][start_pos + 2*j] = triangle[i][j] % 2
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(colors, cmap='RdYlGn')  # Red for even (0), Green for odd (1)
    plt.title("Khayyam's Triangle (Green: Odd, Red: Even)")
    plt.axis('equal')
    plt.show()


# Generate Khayyam's triangle
triangle = generate_khayyam_triangle(n)

# Print Khayyam's triangle, centered
for i, row in enumerate(triangle):
    # Add initial spaces to center the output
    print(" " * (n - i), " ".join(map(str, row)))


generate_point(triangle)
