import numpy as np
import matplotlib.pyplot as plt

def renormalize_block(block):
    """
    Apply renormalization rule to a 2x2 block:
    - If there's a path connecting top to bottom, return 1
    - Otherwise, return 0
    """
    # Check if any column has both cells filled (vertical connection)
    col0_filled = block[0, 0] == 1 and block[1, 0] == 1
    col1_filled = block[0, 1] == 1 and block[1, 1] == 1
    
    # Check for L-shaped paths that connect top to bottom
    path1 = block[0, 0] == 1 and block[1, 1] == 1 and block[1, 0] == 1
    path2 = block[0, 1] == 1 and block[1, 0] == 1 and block[1, 1] == 1
    path3 = block[0, 0] == 1 and block[0, 1] == 1 and block[1, 1] == 1
    path4 = block[0, 0] == 1 and block[0, 1] == 1 and block[1, 0] == 1
    
    # Return 1 if any valid vertical path exists
    if col0_filled or col1_filled or path1 or path2 or path3 or path4:
        return 1
    else:
        return 0

# Examples from the image - top row (should give 1)
examples_true = [
    np.array([[0, 1], [1, 1]]),  # first example
    np.array([[1, 0], [1, 1]]),  # second example
    np.array([[1, 1], [0, 1]]),  # third example
    np.array([[1, 1], [1, 0]]),  # fourth example
    np.array([[1, 0], [1, 0]]),  # fifth example
    np.array([[0, 1], [0, 1]]),  # sixth example
    np.array([[1, 1], [1, 1]]),  # seventh example (all filled)
]

# Examples from the image - bottom row (should give 0)
examples_false = [
    np.array([[1, 0], [0, 0]]),  # top-left only
    np.array([[0, 1], [0, 0]]),  # top-right only
    np.array([[0, 0], [1, 0]]),  # bottom-left only
    np.array([[0, 0], [0, 1]]),  # bottom-right only
    np.array([[0, 0], [1, 1]]),  # bottom row only
    np.array([[1, 1], [0, 0]]),  # top row only
    np.array([[1, 0], [0, 1]]),  # diagonal
    np.array([[0, 1], [1, 0]]),  # other diagonal
    np.array([[0, 0], [0, 0]]),  # all empty
]

# Create a figure to show all examples
fig, axes = plt.subplots(2, len(examples_true), figsize=(15, 6))

# Plot examples that should be True (renormalize to 1)
for i, example in enumerate(examples_true):
    axes[0, i].imshow(example, cmap='Blues', vmin=0, vmax=1)
    axes[0, i].set_title(f"→ {renormalize_block(example)}")
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

# Plot examples that should be False (renormalize to 0)
for i, example in enumerate(examples_false[:len(examples_true)]):
    axes[1, i].imshow(example, cmap='Blues', vmin=0, vmax=1)
    axes[1, i].set_title(f"→ {renormalize_block(example)}")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

plt.tight_layout()
plt.suptitle("Renormalization Examples: Blue=1 (filled), White=0 (empty)", y=1.05)
plt.show()

# Print detailed results for all examples
print("Examples that should renormalize to 1:")
for i, example in enumerate(examples_true):
    print(f"Example {i+1}:\n{example}\nRenormalizes to: {renormalize_block(example)}\n")

print("\nExamples that should renormalize to 0:")
for i, example in enumerate(examples_false):
    print(f"Example {i+1}:\n{example}\nRenormalizes to: {renormalize_block(example)}\n")