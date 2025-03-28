import numpy as np
import matplotlib.pyplot as plt
import random


def explore(current_pos, visited, steps_taken, N):
    if steps_taken == N:
        return 1

    total_paths = 0
    step_vectors = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    ]

    for dx, dy in step_vectors:
        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
        if next_pos not in visited:
            visited.add(next_pos)
            total_paths += explore(next_pos, visited, steps_taken + 1, N)
            visited.remove(next_pos)

    return total_paths


def count_all_SAWs(N):
    start_pos = (0, 0)
    visited = {start_pos}
    return explore(start_pos, visited, 0, N)


for n in range(1, 11):
    count = count_all_SAWs(n)
    print(f"Number of SAWs of length {n}: {count}")
