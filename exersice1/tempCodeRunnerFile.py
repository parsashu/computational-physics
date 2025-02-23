# Print Khayyam's triangle, centered
for i, row in enumerate(triangle):
    # Add initial spaces to center the output
    print(" " * (n - i), " ".join(map(str, row)))