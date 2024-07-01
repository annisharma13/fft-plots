import numpy as np
import matplotlib.pyplot as plt

# Load the data from the files
qz_data = np.loadtxt('WSE003-T.dat')
air_data = np.loadtxt('WSE001-T.dat')

# Extract time and amplitude columns
time_qz, amplitude_qz = qz_data[:, 0], qz_data[:, 1]
time_air, amplitude_air = air_data[:, 0], air_data[:, 1]

# Fourier transformation
sampling_rate_qz = 1 / (time_qz[1] - time_qz[0])  # sampling rate from time intervals
sampling_rate_air = 1 / (time_air[1] - time_air[0])  # sampling rate from time intervals

fft_result_qz = np.fft.fft(amplitude_qz)
fft_result_air = np.fft.fft(amplitude_air)

frequencies_qz = np.fft.fftfreq(len(fft_result_qz), 1 / sampling_rate_qz)
frequencies_air = np.fft.fftfreq(len(fft_result_air), 1 / sampling_rate_air)

# Extract positive frequency components
positive_frequencies_qz = frequencies_qz[:len(frequencies_qz) // 2]
positive_fft_result_qz = fft_result_qz[:len(fft_result_qz) // 2]

positive_frequencies_air = frequencies_air[:len(frequencies_air) // 2]
positive_fft_result_air = fft_result_air[:len(fft_result_air) // 2]

T_calculated = positive_fft_result_qz / positive_fft_result_air
# Select frequencies between 0.5 Hz and 3 Hz
freq_mask = (positive_frequencies_qz >= 0.5) & (positive_frequencies_qz <= 3)
selected_frequencies = positive_frequencies_qz[freq_mask]*10**12
selected_T_calculated = (T_calculated[freq_mask])
print(selected_T_calculated)

# Plotting
plt.figure(figsize=(6, 6))
plt.plot(selected_frequencies[:len(selected_T_calculated)], selected_T_calculated, label='Transmission Coefficient T', color='red')
plt.title('Transmission Coefficient T vs Frequency')
plt.xlabel('Frequency [THz]')
plt.ylabel('Transmission Coefficient T')
plt.legend()
plt.grid(True)
plt.show()
# Save data to a text file
data_to_save = np.column_stack((selected_frequencies[:len(selected_T_calculated)], selected_T_calculated))
np.savetxt('transmission_coefficient.txt', data_to_save, fmt='%.6f', header='Frequency [Hz]  Transmission Coefficient T')

