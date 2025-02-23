# Plot and fill each triangle separately
# plt.fill(left_points[:, 0], left_points[:, 1], alpha=0.3, color='blue', label='Left')
# plt.fill(right_points[:, 0], right_points[:, 1], alpha=0.3, color='red', label='Right')
# plt.fill(top_points[:, 0], top_points[:, 1], alpha=0.3, color='green', label='Top')

# # Plot points for each triangle
# plt.scatter(left_points[:, 0], left_points[:, 1], c='blue', marker='o', s=100)
# plt.scatter(right_points[:, 0], right_points[:, 1], c='red', marker='o', s=100)
# plt.scatter(top_points[:, 0], top_points[:, 1], c='green', marker='o', s=100)

# # Add point numbers for each set
# for i, (x, y) in enumerate(left_points):
#     plt.annotate(f'L{i}', (x, y), xytext=(5, 5), textcoords='offset points')
# for i, (x, y) in enumerate(right_points):
#     plt.annotate(f'R{i}', (x, y), xytext=(5, 5), textcoords='offset points')
# for i, (x, y) in enumerate(top_points):
#     plt.annotate(f'T{i}', (x, y), xytext=(5, 5), textcoords='offset points')