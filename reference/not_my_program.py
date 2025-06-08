import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_c_spectrogram(
    filename,      # path to "spectrogram.bin"
    num_frames,    # number of STFT frames that C wrote
    N,             # FFT length (must match C’s N)
    fs,            # sample rate in Hz (must match C’s wav.sampleRate)
    out_png=None   # if provided, save to this PNG instead of plt.show()
):
    """
    Load a raw float32 spectrogram (num_frames × (N/2) floats) and plot it
    with a frequency axis [0 .. fs/2], matching the pure‐Python version.
    """
    # 1) Load raw data from C
    total_bins = num_frames * (N // 2)
    spec = np.fromfile(filename, dtype=np.float32)
    if spec.size != total_bins:
        raise ValueError(
            f"Expected {total_bins} floats in '{filename}', but found {spec.size}"
        )

    # 2) Reshape into (num_frames, N/2)
    spec = spec.reshape((num_frames, N // 2))

    # 3) Convert to decibels (10·log10), adding a tiny epsilon to avoid log(0)
    eps = 1e-12
    spec_db = 10.0 * np.log10(spec + eps)

    # 4) Transpose so that rows→frequency bins and cols→time frames
    spec_db_plot = spec_db.T   # shape = (N/2, num_frames)

    # 5) Plot with a true frequency‐axis from 0 Hz to fs/2
    plt.figure(figsize=(8, 4))
    # extent = [time_min, time_max, freq_min, freq_max]
    extent = [0, num_frames, 0, fs/2]
    img = plt.imshow(
        spec_db_plot,
        origin='lower',
        aspect='auto',
        cmap='magma',
        extent=extent
    )
    cbar = plt.colorbar(img, label='Power (dB)')
    plt.xlabel('Frame Index (time)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram (C output, magnitude² in dB)')
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=150)
        print(f"Saved spectrogram to '{out_png}'")
    else:
        plt.show()


def main():
    if len(sys.argv) < 5:
        print("Usage: python plot_stft_c_output.py \\")
        print("    <spectrogram.bin> <num_frames> <N> <fs> [out.png]")
        print("")
        print("Example:")
        print("  python plot_stft_c_output.py \\")
        print("      spectrogram.bin 343 1024 44100 spectrogram_c.png")
        sys.exit(1)

    filename   = sys.argv[1]
    num_frames = int(sys.argv[2])
    N          = int(sys.argv[3])
    fs         = int(sys.argv[4])
    out_png    = sys.argv[5] if len(sys.argv) > 5 else None

    plot_c_spectrogram(filename, num_frames, N, fs, out_png)


if __name__ == "__main__":
    main()

