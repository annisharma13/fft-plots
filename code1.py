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

# Plotting original signals
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_qz, amplitude_qz, label='Qz Signal')
plt.plot(time_air, amplitude_air, label='Air Signal', linestyle='dashed')
plt.title('Original Signals')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.xlim(0,8)
# Plotting magnitude spectrum for positive frequencies
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
