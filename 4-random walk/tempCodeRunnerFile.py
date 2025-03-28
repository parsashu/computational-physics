    def explore(current_pos, visited, steps_taken):
        # Base case: if we've taken N steps, we've found a valid path
        if steps_taken == N:
            return 1

        # Try all possible directions
        total_paths = 0
        step_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        for dx, dy in step_vectors:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            # Check if the next position is valid and not visited
            if next_pos not in visited:
                # Add the position to visited set and recurse
                visited.add(next_pos)
                total_paths += explore(next_pos, visited, steps_taken + 1)
                # Backtrack by removing the position from visited
                visited.remove(next_pos)

        return total_paths