#!/usr/bin/env python3
import sys
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

def read_mono_float(wav_path):
    """
    Read a WAV file, convert to mono float32 in [-1, +1].

    Returns:
      fs       : sample rate (int)
      signal_m : 1-D numpy array of type float32, length = num_samples
    """
    fs, data = wavfile.read(wav_path)  # data shape = (num_samples, ) or (num_samples, 2) for stereo
    # Convert to float32 in [-1,+1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)
    # If stereo, average channels
    if data.ndim == 2 and data.shape[1] == 2:
        signal_m = 0.5 * (data[:,0] + data[:,1])
    else:
        signal_m = data.copy()
    return fs, signal_m

def stft_numpy(signal, N=1024, H=512):
    """
    Compute STFT manually in NumPy.

    Parameters:
      signal : 1-D numpy array (mono)
      N      : window length (power of two)
      H      : hop size

    Returns:
      stft_complex : 2-D complex array of shape (num_frames, N)
                     each row = FFT of one windowed frame
    """
    L = signal.shape[0]
    num_frames = 1 + (L - N) // H
    # Precompute Hann window (length N)
    n = np.arange(N)
    hann = 0.5 * (1 - np.cos(2*np.pi * n / (N - 1)))

    # Allocate output
    stft_complex = np.zeros((num_frames, N), dtype=np.complex64)

    for f in range(num_frames):
        start = f * H
        frame = signal[start : start+N]
        # In case the last frame would exceed bounds (shouldn't happen if we computed num_frames this way),
        # you could pad with zeros. Here we assume L >= N + (num_frames - 1)*H exactly.
        windowed = frame * hann
        # In-place FFT of each window:
        X = np.fft.fft(windowed, n=N)
        stft_complex[f, :] = X

    return stft_complex

def magnitude_squared(stft_complex):
    """
    Compute magnitude-squared (power) from a complex STFT array.

    Parameters:
      stft_complex : 2-D complex array of shape (num_frames, N)

    Returns:
      spec_power : 2-D float array of shape (num_frames, N)
    """
    return np.real(stft_complex * np.conj(stft_complex))

def plot_spectrogram(power, fs, N):
    """
    Plot a spectrogram heatmap of the power array.

    Parameters:
      power : 2-D float array, shape = (num_frames, N)
      fs    : sample rate (int)
      N     : FFT length (int)
    """
    # Convert to dB, add small epsilon to avoid log(0)
    eps = 1e-12
    spec_db = 10 * np.log10(power + eps)

    # Transpose so that frequency is vertical axis, time is horizontal
    spec_db = spec_db.T  # now shape = (N, num_frames)

    plt.figure(figsize=(8, 4))
    # extent: [time_min, time_max, freq_min, freq_max]
    num_frames = power.shape[0]
    duration = num_frames * (N//2) / fs  # approximate duration in seconds
    # But for labeling axes, we can use sample indices directly.

    extent = [0, num_frames, 0, fs/2]  # x from frame index 0..num_frames, y from 0..fs/2
    # Only first N//2 bins are unique for real signal; plot up to N//2
    plt.imshow(spec_db[:N//2, :], 
               origin='lower',
               aspect='auto',
               cmap='magma',
               extent=[0, num_frames, 0, fs/2])

    plt.colorbar(label='Power (dB)')
    plt.xlabel('Frame Index')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram (Magnitude Squared, in dB)')
    plt.tight_layout()
    plt.show()
    plt.savefig("reference_spectrogram.png", dpi=150)

def main():
    if len(sys.argv) < 2:
        print("Usage: python python_stft.py path/to/audio.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    fs, signal = read_mono_float(wav_path)
    print(f"Loaded '{wav_path}', sample rate = {fs} Hz, length = {signal.shape[0]} samples")
    
    for i in range(10):
      print(f"Python mon_signal: {signal[i]}")
    # STFT parameters
    N = 1024   # must be power of 2
    H = 512    # hop size (50% overlap)

    # Compute STFT
    stft_complex = stft_numpy(signal, N=N, H=H)
    print(f"Computed STFT: num_frames = {stft_complex.shape[0]}, N = {N}")
   
    for i in range(8):
       real = stft_complex[0,i].real
       imag = stft_complex[0,i].imag
       print(f"Py FFT frame0 bin[{i}]= {real:.6f} + {imag:.6f}i") 
    spec_power = magnitude_squared(stft_complex)
	
    for i in range(5):
       print("Python spec:" )
       print (spec_power[i])
    # Plot as heatmap
    plot_spectrogram(spec_power, fs, N)


if __name__ == "__main__":
    main()

