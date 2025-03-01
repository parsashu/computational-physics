import matplotlib.pyplot as plt
import numpy as np


# Specify the number of rows for Khayyam's triangle
n = 1000


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
    width = len(triangle[-1])

    # Create a matrix to store the colors (1 for odd, 0 for even)
    colors = np.zeros((height, width))

    # Fill the matrix with 1s (odd) and 2s (even) in a triangular pattern
    for i in range(height):
        # Calculate starting position for this row to center it
        start_pos = (width - i) // 2
        for j in range(len(triangle[i])):
            # Use 2 for even numbers, 1 for odd numbers
            colors[i][start_pos + j] = 2 if triangle[i][j] % 2 == 0 else 1

    # Create a masked array where 0s are masked
    masked_colors = np.ma.masked_where(colors == 0, colors)

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(
        masked_colors, cmap="RdYlGn"
    )  # Red for even (2), Green for odd (1), transparent for 0
    plt.title(f"Khayyam's Triangle (n={height}) (Green: Odd, Red: Even)")
    plt.axis("equal")
    plt.show()


# Generate Khayyam's triangle
triangle = generate_khayyam_triangle(n)


generate_point(triangle)
