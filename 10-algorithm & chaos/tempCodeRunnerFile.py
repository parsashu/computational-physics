def phase_space_distance(x_true, v_true, x_sim, v_sim):
    return np.sqrt((x_true - x_sim)**2 + (v_true - v_sim)**2)