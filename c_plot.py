import numpy as np
import matplotlib.pyplot as plt

# 1) Parameters—match these to your C code
num_frames = 374     # replace with the exact num_frames printed by your C program
N = 1024            # FFT length used in C
fs = 48000           # sample rate of your WAV file (replace if different)

# 2) Load the raw binary file (float32, row‐major: frame0 bins[0..N-1], frame1 bins[N..2N-1], etc.)
spec = np.fromfile("spectrogram.bin", dtype=np.float32)
print("raw.bin length =",spec.size," expected =",num_frames * N)
spec = spec.reshape((num_frames, N))

# 3) Keep only the first N//2 bins (positive frequencies 0…fs/2)
half = N // 2
spec_half = spec[:, :half]       # shape = (num_frames, N//2)

# 4) Convert to decibels for dynamic range (avoid log(0) with a small epsilon)
spec_db = 10 * np.log10(spec_half + 1e-12)

# 5) Transpose so that rows → frequency, columns → time
spec_db_plot = spec_db.T         # shape = (N//2, num_frames)

# 6) Plot with a true frequency axis in Hz
plt.figure(figsize=(6, 4))

# extent = [time_min, time_max, freq_min, freq_max]
extent = [0, num_frames, 0, fs/2]

plt.imshow(
    spec_db_plot,
    origin='lower',
    aspect='auto',
    cmap='magma',
    extent=extent
)

plt.colorbar(label='Power (dB)')
plt.xlabel('Frame Index (time)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram (log‐power)')
plt.tight_layout()
# plt.show()
plt.savefig("spectrogram_hz.png", dpi=150)

