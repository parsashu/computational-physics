# Define the power law function: w(t) = A * t^beta
def power_law(t, A, beta):
    return A * t**beta


# Use only the latter part of the data for fitting (after initial transient)
fit_start = time // 10  # Start fitting from 10% of the data
x_data = np.arange(fit_start, time)
y_data = w_array[fit_start:]

# Perform the curve fitting
params, covariance = curve_fit(power_law, x_data, y_data)
A_fit, beta_fit = params
beta_error = np.sqrt(np.diag(covariance))[1]  # Extract the error in beta

# Generate the fitted curve
y_fit = power_law(x_data, A_fit, beta_fit)

# Plot the fitted curve
plt.plot(x_data, y_fit, "b--", label=f"Fitted: t^{beta_fit:.3f}±{beta_error:.3f}")
plt.legend()

print(f"Fitted growth exponent (beta): {beta_fit:.4f} ± {beta_error:.4f}")
print(f"Amplitude (A): {A_fit:.6e}")
plt.show()