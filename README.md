# fft-plots
This is first code.
Certainly! Let's break down the provided code step by step:

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

In summary, the code loads time-domain signals for quartz and air, computes their FFT to transform them into the frequency domain, and plots both the original time-domain signals and their frequency-domain magnitude spectra.
