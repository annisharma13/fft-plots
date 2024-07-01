import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#Data  
data = np.loadtxt('filtered_T_values_upto3.txt')

frequencies = data[:, 0] * 1e12  # Convert THz to Hz
real_T_calculated = data[:, 1]
imaginary_T_calculated = data[:, 2]

c = 3e8  # Speed of light in m/s
l = 1e-3  # Thickness of the sample in m

# Function to calculate T_analytical
def T_analytical(ns, w, l, c):
    term1 = (ns + 1)**2 / (2 * ns)
    cos_term = np.cos((ns - 1) * w * l / c)
    sin_term = np.sin((ns - 1) * w * l / c)
    return term1 * (cos_term - 1j * sin_term)

# Function to solve for ns
def equations(ns, real_T_calculated, imaginary_T_calculated, w, l, c):
    ns_complex = ns[0] + 1j * ns[1]
    T_anal = T_analytical(ns_complex, w, l, c)
    real_part_error = T_anal.real - real_T_calculated
    imaginary_part_error = T_anal.imag - imaginary_T_calculated
    return [real_part_error, imaginary_part_error]

# Solve for ns for each data point
ns_values = []
for f, real_T, imag_T in zip(frequencies, real_T_calculated, imaginary_T_calculated):
    w = 2 * np.pi * f
    ns_guess = [2.5,-2.5] # Initial guess
    ns_solution = fsolve(equations, ns_guess, args=(real_T, imag_T, w, l, c))
    ns_values.append(ns_solution)

ns_values = np.array(ns_values)
real_ns = ns_values[:, 0]
imaginary_ns = ns_values[:, 1]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(frequencies / 1e12, real_ns, label='Real part of $n_s$')
plt.plot(frequencies / 1e12, imaginary_ns, label='Imaginary part of $n_s$')
plt.xlabel('Frequency (THz)')
plt.ylabel('Refractive Index $n_s$')
plt.title('Refractive Index vs Frequency')
plt.legend()
plt.grid(True)
plt.show()
