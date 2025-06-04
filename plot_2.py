import numpy as np
import matplotlib.pyplot as plt

num_frames = 374  # ← exactly what C printed
N = 1024
fs = 44100       # ← same sample rate you read via dr_wav

# 1) load the full 1024‐bin spectrogram
spec = np.fromfile("spectrogram.bin", dtype=np.float32)
spec = spec.reshape((num_frames, N))

# 2) keep only bins 0..511 (positive frequencies)
half = N // 2
spec_half = spec[:, :half]               # shape = (num_frames, 512)

# 3) convert to dB
spec_db = 10 * np.log10(spec_half + 1e-12)

# 4) transpose so that rows→frequency, cols→time
spec_db_plot = spec_db.T                 # shape = (512, num_frames)

# 5) plot with 0…fs/2 on y‐axis
plt.figure(figsize=(6, 4))
plt.imshow(
    spec_db_plot,
    origin='lower',
    aspect='auto',
    cmap='magma',
    extent=[0, num_frames, 0, fs/2]
)
plt.colorbar(label='Power (dB)')
plt.xlabel('Frame Index (time)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram (C output, mapped to 0..fs/2)')
plt.tight_layout()
# plt.show()
plt.savefig("spectrogram_hz.png", dpi=150)

