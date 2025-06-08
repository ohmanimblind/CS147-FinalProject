#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda.h>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"    // single‐header WAV loader

// ----------------------------------------
// Simple single‐precision complex type
// ----------------------------------------
typedef struct {
    float real;
    float imag;
} Complex;

// ----------------------------------------
// Device‐side helpers for complex arithmetic
// ----------------------------------------
__device__ __forceinline__ Complex complex_add(const Complex &a, const Complex &b) {
    return { a.real + b.real,  a.imag + b.imag };
}

__device__ __forceinline__ Complex complex_sub(const Complex &a, const Complex &b) {
    return { a.real - b.real,  a.imag - b.imag };
}

__device__ __forceinline__ Complex complex_mult(const Complex &a, const Complex &b) {
    return {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

// ----------------------------------------
// GPU kernel: compute batched STFT (radix‐2 in shared memory)
// Each block → one frame (length N). Each thread t ∈ [0..N-1] → one sample/bin.
// Shared memory holds N Complex values per block.
// ----------------------------------------
extern "C"
__global__ void stft_kernel(const float* d_signal,
                            const float* d_hann,
                            Complex*      d_stft_out,
                            int           totalSamples,
                            int           N,
                            int           H,
                            int           num_frames)
{
    // Which STFT frame (block index) and which thread in that block
    int frame = blockIdx.x;    // [0 .. num_frames-1]
    int t     = threadIdx.x;   // [0 .. N-1]

    // Bounds check (each block handles exactly one frame, each thread one bin)
    if (frame >= num_frames || t >= N) return;

    extern __shared__ Complex s_data[]; 
    // s_data[t] will hold the windowed sample (as real + imag=0) in stage 1,
    // and then holds the in-place FFT buffer for all stages.

    // 1) SLICE & WINDOW:
    //    Each frame starts at sample index "frame * H".
    //    Thread t loads sample at index (frame*H + t), multiplies by d_hann[t].
    int start = frame * H;     // sample index where this frame begins
    int idx   = start + t;     // global sample index for this thread
    float windowed = 0.0f;
    if (idx < totalSamples) {
        windowed = d_signal[idx] * d_hann[t];
    }
    s_data[t].real = windowed;
    s_data[t].imag = 0.0f;
    __syncthreads();

    // 2) IN-PLACE iterative Radix-2 FFT on s_data[0..N-1]
    //    We assume N is a power of 2. Compute log2(N) via __ffs(N)-1.
    int logN = __ffs(N) - 1;  // valid only if N is power of two

    for (int s = 1; s <= logN; ++s) {
        int m     = 1 << s;       // m = 2, 4, 8, …, N
        int halfM = m >> 1;       // halfM = m/2

        // k = position inside this half-block
        int k = t & (halfM - 1);

        // Twiddle factor: W_m^k = exp(−2π i * k / m)
        float angle = -2.0f * M_PI * k / float(m);
        Complex w   = { cosf(angle), sinf(angle) };

        // Determine group start index in shared memory:
        int groupStart = (t >> s) * m;   // (t / m) * m
        int index1     = groupStart + k; // top of butterfly
        int index2     = index1 + halfM; // bottom of butterfly

        Complex u = s_data[index1];
        Complex v = s_data[index2];
        Complex v_tw = complex_mult(v, w); 
        __syncthreads();

        s_data[index1] = complex_add(u, v_tw);
        s_data[index2] = complex_sub(u, v_tw);
        __syncthreads();
    }

    // 3) WRITE final FFT result to global memory
    d_stft_out[ frame * N + t ] = s_data[t];
}

// ----------------------------------------
// GPU kernel: compute magnitude‐squared (power) for only first N/2 bins
//  Each thread handles exactly one bin of one frame: idx = frame * (N/2) + b
// ----------------------------------------
extern "C"
__global__ void spectrogram_kernel(const Complex* d_stft,
                                   float*         d_power,
                                   int            num_frames,
                                   int            N_half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalBins = num_frames * N_half;
    if (idx >= totalBins) return;

    // Extract real & imag from the positive-frequency half (bins 0..N/2-1)
    float re = d_stft[idx].real;
    float im = d_stft[idx].imag;
    d_power[idx] = re*re + im*im;
}

// ----------------------------------------
// Host function: wrapper to launch STFT kernel
// ----------------------------------------
void launch_stft(const float* d_signal,
                 const float* d_hann,
                 Complex*      d_stft_out,
                 int           totalSamples,
                 int           N,
                 int           H,
                 int           num_frames)
{
    int shared_bytes = N * sizeof(Complex);
    dim3 grid (num_frames);
    dim3 block(N);
    stft_kernel<<< grid, block, shared_bytes >>>(
        d_signal, d_hann, d_stft_out,
        totalSamples, N, H, num_frames
    );
}

// ----------------------------------------
// Host function: wrapper to launch spectrogram (power) kernel
//    We only compute bins 0..(N/2−1) because the others are mirror.
// ----------------------------------------
void launch_spectrogram(const Complex* d_stft,
                        float*         d_power,
                        int            num_frames,
                        int            N_half)
{
    int totalBins = num_frames * N_half;
    int blockSize = 256;
    int gridSize  = (totalBins + blockSize - 1) / blockSize;
    spectrogram_kernel<<< gridSize, blockSize >>>(
        d_stft, d_power, num_frames, N_half
    );
}

// ----------------------------------------
// Entry point: read WAV, copy to GPU, launch STFT + spectrogram, write output
// ----------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s path/to/audio.wav\n", argv[0]);
        return 1;
    }
    const char* wav_path = argv[1];

    // --------- 1) Load WAV via dr_wav  ----------
    drwav wav;
    if (!drwav_init_file(&wav, wav_path, NULL)) {
        printf("Failed to open WAV file: %s\n", wav_path);
        return 1;
    }
    uint32_t fs = wav.sampleRate;
    uint64_t totalFrames = wav.totalPCMFrameCount; 
    int channels = wav.channels;
    printf("Opened WAV: %s\n", wav_path);
    printf("  Sample Rate: %u Hz\n", fs);
    printf("  Channels:    %u\n", (unsigned)channels);
    printf("  Total Frames(per channel): %llu\n", (unsigned long long)totalFrames);

    // Read all PCM frames (interleaved if stereo) as int16
    uint64_t totalSamples = totalFrames * channels;
    int16_t* buffer = new int16_t[ totalSamples ];
    if (!buffer) {
        printf("Failed to allocate PCM buffer\n");
        drwav_uninit(&wav);
        return 1;
    }
    uint64_t framesRead = drwav_read_pcm_frames_s16(
        &wav, totalFrames, buffer
    );
    if (framesRead != totalFrames) {
        printf("Warning: requested %llu frames, but read %llu\n",
               (unsigned long long)totalFrames,
               (unsigned long long)framesRead);
    }

    // --------- 2) Convert to mono float32 in [-1,+1] ----------
    float* h_mono = new float[ totalFrames ];
    if (!h_mono) {
        printf("Failed to allocate mono array\n");
        delete[] buffer;
        drwav_uninit(&wav);
        return 1;
    }
    for (uint64_t i = 0; i < totalFrames; ++i) {
        if (channels == 1) {
            h_mono[i] = buffer[i] / 32768.0f;
        } else {
            int16_t L = buffer[i*2 + 0];
            int16_t R = buffer[i*2 + 1];
            h_mono[i] = ( (float)L + (float)R ) / (2.0f * 32768.0f);
        }
    }
    // (Optional) print first few samples for sanity
    printf("C raw mono[0..7]: ");
    for (int i = 0; i < 8; ++i) {
        printf("%f ", h_mono[i]);
    }
    printf("\n");

    delete[] buffer;
    drwav_uninit(&wav);

    // --------- 3) STFT parameters & Hann window ----------
    const int N = 1024;            // FFT length (power of two)
    const int H = 512;             // Hop size (50% overlap)
    if ((int)totalFrames < N) {
        printf("Audio length (%d) is shorter than window (%d)\n",
               (int)totalFrames, N);
        delete[] h_mono;
        return 1;
    }
    int num_frames = ((int)totalFrames - N) / H + 1;
    printf("STFT: N=%d, H=%d, num_frames=%d\n", N, H, num_frames);

    // Allocate & compute Hann window on host
    float* h_hann = new float[N];
    for (int i = 0; i < N; ++i) {
        h_hann[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (float)(N - 1)));
    }

    // --------- 4) Allocate GPU memory and copy inputs ----------
    float*   d_signal     = nullptr;  // mono audio on device
    float*   d_hann       = nullptr;  // Hann window on device
    Complex* d_stft_full  = nullptr;  // full N‐bin FFT output on device
    float*   d_power_half = nullptr;  // N/2‐bin power output on device

    size_t bytes_signal   = totalFrames * sizeof(float);
    size_t bytes_hann     = N * sizeof(float);
    size_t bytes_stft     = (size_t)num_frames * N * sizeof(Complex);
    size_t bytes_power    = (size_t)num_frames * (N/2) * sizeof(float);

    // 4.1) Allocate & copy mono signal to device
    cudaError_t err = cudaMalloc((void**)&d_signal, bytes_signal);
    if (err != cudaSuccess) {
        printf("cudaMalloc d_signal failed: %s\n", cudaGetErrorString(err));
  
    }
    err = cudaMemcpy(d_signal, h_mono, bytes_signal, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy h_mono→d_signal failed: %s\n", cudaGetErrorString(err));
   
    }

    // 4.2) Allocate & copy Hann window to device
    err = cudaMalloc((void**)&d_hann, bytes_hann);
    if (err != cudaSuccess) {
        printf("cudaMalloc d_hann failed: %s\n", cudaGetErrorString(err));
   
    }
    err = cudaMemcpy(d_hann, h_hann, bytes_hann, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy h_hann→d_hann failed: %s\n", cudaGetErrorString(err));
       
    }

    // 4.3) Allocate full‐spectrum STFT output buffer
    err = cudaMalloc((void**)&d_stft_full, bytes_stft);
    if (err != cudaSuccess) {
        printf("cudaMalloc d_stft_full failed: %s\n", cudaGetErrorString(err));
     
    }

    // 4.4) Allocate half‐spectrum power output buffer
    err = cudaMalloc((void**)&d_power_half, bytes_power);
    if (err != cudaSuccess) {
        printf("cudaMalloc d_power_half failed: %s\n", cudaGetErrorString(err));
       
    }

    // --------- 5) Launch STFT kernel ----------
    launch_stft(d_signal, d_hann, d_stft_full,
                (int)totalFrames, N, H, num_frames);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("stft_kernel launch failed: %s\n", cudaGetErrorString(err));
   
    }

    // --------- 6) Launch spectrogram (power) kernel on only first N/2 bins ----------
    int N_half = N / 2;
    launch_spectrogram(d_stft_full, d_power_half, num_frames, N_half);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("spectrogram_kernel launch failed: %s\n", cudaGetErrorString(err));
       
    }

    // --------- 7) Copy result back to host and write binary file ----------
    float* h_power = (float*)malloc(bytes_power);
    if (!h_power) {
        printf("Failed to allocate host spectrogram array\n");

    }
    err = cudaMemcpy(h_power, d_power_half, bytes_power, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy d_power_half→h_power failed: %s\n", cudaGetErrorString(err));
        free(h_power);
    }

    // Write out a raw binary file: num_frames * (N/2) floats
    FILE* fout = fopen("spectrogram.bin", "wb");
    if (!fout) {
        printf("Failed to open spectrogram.bin for writing\n");
        free(h_power);
       
    }
    fwrite(h_power, sizeof(float), (size_t)num_frames * N_half, fout);
    fclose(fout);
    printf("Wrote spectrogram.bin  (%d frames × %d bins)\n", num_frames, N_half);
    free(h_power);


    // ------------- Free GPU and host resources -------------
    if (d_signal)     cudaFree(d_signal);
    if (d_hann)       cudaFree(d_hann);
    if (d_stft_full)  cudaFree(d_stft_full);
    if (d_power_half) cudaFree(d_power_half);

    delete[] h_hann;
    delete[] h_mono;

    return 0;
}

