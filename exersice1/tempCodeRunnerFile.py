    # # Fill the matrix with 1s (odd) and 0s (even) in a triangular pattern
    # for i in range(height):
    #     # Calculate starting position for this row to center it
    #     start_pos = (width - (2 * i + 1)) // 2
    #     for j in range(len(triangle[i])):
    #         colors[i][start_pos + 2*j] = triangle[i][j] % 2