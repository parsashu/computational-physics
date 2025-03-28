import matplotlib.pyplot as plt


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


lengths = list(range(1, 15))
counts = []
counts_by_free = []

for n in lengths:
    count = count_all_SAWs(n)
    counts.append(count)

    count_by_free = count / (4**n)
    counts_by_free.append(count_by_free)


plt.figure(figsize=(10, 6))
plt.plot(lengths, counts, "o-", linewidth=2, markersize=8)
plt.xlabel("Length (N)")
plt.ylabel("Number of SAWs")
plt.title("Number of Self-Avoiding Walks vs Length")
plt.grid(True)
plt.xticks(lengths)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(lengths, counts_by_free, "o-", linewidth=2, markersize=8)
plt.xlabel("Length (N)")
plt.ylabel("Count / 4^N")
plt.title("Ratio of SAWs to Free Walks vs Length")
plt.grid(True)
plt.xticks(lengths)
plt.show()

