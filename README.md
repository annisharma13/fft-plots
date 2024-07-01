# fft-plots
This is first code.
code 1 and t calculated code can be coded together, here's the explaination of both codes combined and treated as code first.
### 1. Loading Data

```python
qz_data = np.loadtxt('WSE003-T.dat')
air_data = np.loadtxt('WSE001-T.dat')
```

- `np.loadtxt` is used to load data from text files `WSE003-T.dat` and `WSE001-T.dat`.
- These files presumably contain two columns: time and amplitude.

### 2. Extracting Time and Amplitude Columns

```python
time_qz, amplitude_qz = qz_data[:, 0], qz_data[:, 1]
time_air, amplitude_air = air_data[:, 0], air_data[:, 1]
```

- This splits the loaded data into `time_qz` and `amplitude_qz` for the quartz (Qz) signal, and `time_air` and `amplitude_air` for the air signal.
- `qz_data[:, 0]` extracts the first column (time), and `qz_data[:, 1]` extracts the second column (amplitude). The same applies to `air_data`.

### 3. Fourier Transformation

```python
sampling_rate_qz = 1 / (time_qz[1] - time_qz[0])
sampling_rate_air = 1 / (time_air[1] - time_air[0])
```

- The sampling rate is calculated based on the time intervals between consecutive samples. This is done by taking the inverse of the difference between the first two time points.

```python
fft_result_qz = np.fft.fft(amplitude_qz)
fft_result_air = np.fft.fft(amplitude_air)
```

- `np.fft.fft` computes the Fast Fourier Transform (FFT) of the amplitude data. FFT transforms the time-domain signal into the frequency domain.

```python
frequencies_qz = np.fft.fftfreq(len(fft_result_qz), 1 / sampling_rate_qz)
frequencies_air = np.fft.fftfreq(len(fft_result_air), 1 / sampling_rate_air)
```

- `np.fft.fftfreq` generates the frequency bins corresponding to the FFT results.

### 4. Extracting Positive Frequency Components

```python
positive_frequencies_qz = frequencies_qz[:len(frequencies_qz) // 2]
positive_fft_result_qz = fft_result_qz[:len(fft_result_qz) // 2]
positive_frequencies_air = frequencies_air[:len(frequencies_air) // 2]
positive_fft_result_air = fft_result_air[:len(fft_result_air) // 2]
```

- FFT results are symmetric, so we only need the first half (positive frequencies). This is done by slicing the arrays.

### 5. Plotting the Original Signals

```python
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_qz, amplitude_qz, label='Qz Signal')
plt.plot(time_air, amplitude_air, label='Air Signal', linestyle='dashed')
plt.title('Original Signals')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.xlim(0,8)
```

- A figure with a size of 12x6 inches is created.
- The first subplot is created to plot the original time-domain signals for both quartz and air.
- `plt.plot` is used to plot time vs. amplitude for both signals. `label` is used to provide a legend, and `linestyle='dashed'` differentiates the air signal.
- `plt.xlim(0,8)` limits the x-axis from 0 to 8 seconds.

### 6. Plotting the Magnitude Spectrum for Positive Frequencies

```python
plt.subplot(2, 1, 2)
plt.plot(positive_frequencies_qz, np.abs(positive_fft_result_qz), label='Qz Signal')
plt.plot(positive_frequencies_air, np.abs(positive_fft_result_air), label='Air Signal', linestyle='dashed')
plt.title('Fourier Transform (Positive Frequencies)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.legend()
plt.xlim(0,8)
plt.tight_layout()
plt.show()
```

- The second subplot is created to plot the magnitude spectrum of the positive frequency components.
- `np.abs` is used to get the magnitude of the FFT results (since FFT results are complex numbers).
- The plot shows frequency (x-axis) vs. magnitude (y-axis) for both signals, with a legend to differentiate them.
- `plt.tight_layout()` adjusts the subplots to fit in the figure area neatly.
- `plt.show()` displays the figure.


### 7. Calculate the Transmission Coefficient

```python
T_calculated = positive_fft_result_qz / positive_fft_result_air
```
- The transmission coefficient `T` is calculated by dividing the FFT result of the quartz sample by that of the air sample.

### 8. Select Frequencies Between 0.5 Hz and 3 Hz

```python
freq_mask = (positive_frequencies_qz >= 0.5) & (positive_frequencies_qz <= 3)
selected_frequencies = positive_frequencies_qz[freq_mask] * 10**12
selected_T_calculated = T_calculated[freq_mask]
```
- A mask is created to filter frequencies between 0.5 Hz and 3 Hz.
- The corresponding frequencies and transmission coefficients are selected and converted to THz (terahertz).

### 9. Plot the Results

```python
plt.figure(figsize=(6, 6))
plt.plot(selected_frequencies[:len(selected_T_calculated)], selected_T_calculated, label='Transmission Coefficient T', color='red')
plt.title('Transmission Coefficient T vs Frequency')
plt.xlabel('Frequency [THz]')
plt.ylabel('Transmission Coefficient T')
plt.legend()
plt.grid(True)
plt.show()
```
- A plot is created with the selected frequencies on the x-axis and the transmission coefficient on the y-axis.
- The plot is labeled and displayed.

### 10. Save Data to a Text File

```python
data_to_save = np.column_stack((selected_frequencies[:len(selected_T_calculated)], selected_T_calculated))
np.savetxt('transmission_coefficient.txt', data_to_save, fmt='%.6f', header='Frequency [Hz]  Transmission Coefficient T')
```
- The selected frequencies and transmission coefficients are combined into a single array.
- This data is saved to a text file named `transmission_coefficient.txt` with a header and specified format.

### Summary

This script loads time-domain data, performs a Fourier transform to convert it into the frequency domain, calculates the transmission coefficient, filters the data to a specific frequency range, plots the results, and saves the data to a text file.

this is our final code
Let's break down and explain the provided code in detail, step by step.

### 1 Importing Libraries

```python
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
```

Here, we import necessary libraries:
- `numpy` for numerical operations.
- `fsolve` from `scipy.optimize` to solve systems of nonlinear equations.
- `matplotlib.pyplot` for plotting the results.

### 2 Loading and Preparing Data

```python
data = np.loadtxt('transmission_coefficient.txt')

frequencies = data[:, 0] * 1e12  # Convert THz to Hz
real_T_calculated = data[:, 1]
imaginary_T_calculated = data[:, 2]
```

We load the data from a text file named `'filtered_T_values_upto3.txt'`. The data is assumed to have three columns:
1. Frequency in THz.
2. Real part of the calculated transmission coefficient \( T \).
3. Imaginary part of the calculated transmission coefficient \( T \).

The frequency values are converted from THz to Hz by multiplying by \( 1 \times 10^{12} \).

### 3 Constants

```python
c = 3e8  # Speed of light in m/s
l = 1e-3  # Thickness of the sample in m
```

We define two constants:
- `c`: Speed of light in vacuum (3 x 10^8 m/s).
- `l`: Thickness of the sample (1 mm).

### 4 Function to Calculate Analytical Transmission Coefficient

```python
def T_analytical(ns, w, l, c):
    term1 = (ns + 1)**2 / (2 * ns)
    cos_term = np.cos((ns - 1) * w * l / c)
    sin_term = np.sin((ns - 1) * w * l / c)
    return term1 * (cos_term - 1j * sin_term)
```

This function calculates the analytical transmission coefficient \( T_{\text{analytical}} \) for a given refractive index `ns`, angular frequency `w`, sample thickness `l`, and speed of light `c`. It uses the formula:

\[ T_{\text{analytical}} = \frac{(n_s + 1)^2}{2n_s} \left( \cos \left( \frac{(n_s - 1) w l}{c} \right) - i \sin \left( \frac{(n_s - 1) w l}{c} \right) \right) \]

### 5 Function to Solve for Refractive Index

```python
def equations(ns, real_T_calculated, imaginary_T_calculated, w, l, c):
    ns_complex = ns[0] + 1j * ns[1]
    T_anal = T_analytical(ns_complex, w, l, c)
    real_part_error = T_anal.real - real_T_calculated
    imaginary_part_error = T_anal.imag - imaginary_T_calculated
    return [real_part_error, imaginary_part_error]
```

This function defines the system of equations to solve for the refractive index `ns` by matching the real and imaginary parts of the calculated transmission coefficient with the analytical transmission coefficient. The function returns the difference (error) between the real and imaginary parts of \( T_{\text{analytical}} \) and \( T_{\text{calculated}} \).

### 6 Solving for Refractive Index for Each Data Point

```python
ns_values = []
for f, real_T, imag_T in zip(frequencies, real_T_calculated, imaginary_T_calculated):
    w = 2 * np.pi * f
    ns_guess = [2.5, -2.5]  # Initial guess
    ns_solution = fsolve(equations, ns_guess, args=(real_T, imag_T, w, l, c))
    ns_values.append(ns_solution)

ns_values = np.array(ns_values)
real_ns = ns_values[:, 0]
imaginary_ns = ns_values[:, 1]
```

In this loop, we iterate over each frequency and its corresponding real and imaginary parts of the transmission coefficient. For each data point:
1. We convert the frequency to angular frequency `w`.
2. We use `fsolve` to solve the system of equations for the refractive index `ns`, starting with an initial guess of `[2.5, -2.5]`.
3. The solution `ns_solution` is appended to the list `ns_values`.

After the loop, we convert `ns_values` to a NumPy array and separate the real and imaginary parts of the refractive index into `real_ns` and `imaginary_ns`.

### 7 Plotting the Results

```python
plt.figure(figsize=(10, 6))
plt.plot(frequencies / 1e12, real_ns, label='Real part of $n_s$')
plt.plot(frequencies / 1e12, imaginary_ns, label='Imaginary part of $n_s$')
plt.xlabel('Frequency (THz)')
plt.ylabel('Refractive Index $n_s$')
plt.title('Refractive Index vs Frequency')
plt.legend()
plt.grid(True)
plt.show()
```

Finally, we plot the real and imaginary parts of the refractive index against the frequency (converted back to THz for better readability). The plot includes labels, a title, a legend, and a grid for better visualization.

### Summary

The code:
1. Loads the data from a file.
2. Converts frequencies to Hz.
3. Defines the analytical model for the transmission coefficient.
4. Defines a function to solve for the refractive index.
5. Iterates over the data to find the refractive index for each frequency.
6. Plots the real and imaginary parts of the refractive index as functions of frequency.

This process allows us to analyze how the refractive index varies with frequency based on the transmission coefficient data.




