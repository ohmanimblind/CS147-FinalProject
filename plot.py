import numpy as np
import matplotlib.pyplot as plt

# 1) Parameters—make sure these match your C++ code!
num_frames = 374      # for example
N = 1024              # FFT length

# 2) Load the raw binary file
# Each entry is a float32, row-major: frame 0 bins [0..N-1], frame 1 bins [N..2N-1], etc.
spec = np.fromfile("spectrogram.bin", dtype=np.float32)

# 3) Reshape into (num_frames, N)
spec = spec.reshape((num_frames, N))

# 4) (Optional) Convert to decibels for better dynamic range:
# Avoid log(0) by adding a small epsilon
spec_db = 10 * np.log10(spec + 1e-12)

# 5) Plot the heatmap
plt.figure(figsize=(6, 4))

# By default, imshow will put row 0 at the top; we often want "time→right" and "low freqs at bottom".
# So we transpose and use origin='lower'
plt.imshow(spec_db.T,     # now shape (N, num_frames)
           origin='lower',
           aspect='auto',
           interpolation='nearest',
           cmap='magma')    # any colormap you like: 'viridis', 'inferno', 'magma', etc.

plt.colorbar(label='Power (dB)')
plt.xlabel('Frame Index (time)')
plt.ylabel('Frequency Bin Index')
plt.title('Spectrogram (log‐power)')

# If you know your sample rate fs, you can also convert "bin index" → frequency in Hz:
#    freqs = np.linspace(0, fs/2, N)   # for a real‐input FFT, only the first N/2 are unique
# and replace ylabel and ticks accordingly.

plt.tight_layout()
#plt.show()
plt.savefig("spectrogram.png",dpi=150)
