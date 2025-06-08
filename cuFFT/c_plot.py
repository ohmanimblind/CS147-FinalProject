import numpy as np
import matplotlib.pyplot as plt


num_frames = 223    # always match with num_frames
N = 1024            # FFT length used in C
fs = 44100           # sample rate of your WAV file (replace if different)

spec = np.fromfile("spectrogram.bin", dtype=np.float32)
print("raw.bin length =",spec.size," expected =",num_frames * N)
spec = spec.reshape((num_frames, N))

half = N // 2
spec_half = spec[:, :half]       # shape = (num_frames, N//2)

spec_db = 10 * np.log10(spec_half + 1e-12)

spec_db_plot = spec_db.T         # shape = (N//2, num_frames)

plt.figure(figsize=(6, 4))

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

