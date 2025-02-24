def tail_func(points):
    points_copy = points.copy()
    rotated = rotate_shape(points_copy, top_angle)
    scaled = scale_shape(rotated, 0.9, 0.1)
    return scaled