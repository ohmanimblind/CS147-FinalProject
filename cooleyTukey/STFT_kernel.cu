#include <cuda.h>
#include <math.h>
#define PI  3.14159265358979323846f
#define BLOCK_SIZE 256
	//<<< >>>;
	//fftkernel<<< >>>;

   typedef struct{
       float real;
      float imag;
  }Complex;
  
  __device__ __forceinline__ Complex complex_add(Complex a, Complex b){
          return (Complex){a.real + b.real, a.imag + b.imag};
  }
  __device__ __forceinline__ Complex complex_sub(Complex a, Complex b){
          return(Complex){a.real - b.real, a.imag - b.imag};
  }
  
  __device__ __forceinline__ Complex complex_mult(Complex a, Complex b){
         return(Complex){a.real*b.real-a.imag*b.imag, a.imag*b.real+b.imag*a.real};
  }
  
  
 __global__ void spectrogram(const Complex* d_stft, float* d_spectro, int totalBins){
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if(t >= totalBins) return;	

	float re = d_stft[t].real;
	float im = d_stft[t].imag;

	d_spectro[t] = re * re + im * im;

} 

void make_spectro(const Complex* d_stft, float* d_spectro, int num_frames, int N){
	int total_Bins = num_frames * N;
	dim3 block_size(BLOCK_SIZE);
	dim3 grid = (total_Bins + BLOCK_SIZE - 1) / BLOCK_SIZE;
	spectrogram<<<grid, BLOCK_SIZE >>>(d_stft,d_spectro, total_Bins);
} 



static __device__ __forceinline__ int bit_reverse(int x, int logN) {
    // Reverse the lowest logN bits of x
    int y = 0;
    for (int i = 0; i < logN; ++i) {
        y = (y << 1) | ( (x >> i) & 1 );
    }
    return y;
}
extern "C"
__global__ void stft_kernel_fixed(
    const float* d_signal,
    const float* d_hann,
    Complex*      d_stft_out,
    int           totalSamples,
    int           N,
    int           H,
    int           num_frames
) {
    int frame = blockIdx.x;
    int t     = threadIdx.x;
    if (frame >= num_frames || t >= N) return;

    extern __shared__ Complex s_data[];

    // 1) load & window
    int start = frame * H;
    int idx   = start + t;
    float windowed = 0.0f;
    if (idx < totalSamples) {
        windowed = d_signal[idx] * d_hann[t];
    }
    s_data[t].real = windowed;
    s_data[t].imag = 0.0f;
    __syncthreads();

    // 2) BIT‐REVERSAL reorder in‐place
    int logN = __ffs(N) - 1;  // log2(N)
    int rev  = bit_reverse(t, logN);
    if (rev > t) {
        // swap s_data[t] and s_data[rev]
        Complex tmp = s_data[t];
        s_data[t] = s_data[rev];
        s_data[rev] = tmp;
    }
    __syncthreads();

    // 3) Iterative Radix‐2 FFT (standard butterflies)
    for (int s = 1; s <= logN; ++s) {
        int m     = 1 << s; // 2^m
        int halfM = m >> 1; // m / 2
        int k     = t & (halfM - 1); //% mod operation for finding butterfly index

        float angle = -2.0f * M_PI * k / float(m);
        Complex w = { cosf(angle), sinf(angle) }; //twindle factor

        int groupStart = (t >> s) * m; //determines block start
        int index1 = groupStart + k;   // "top" of butterfly pair
        int index2 = index1 + halfM;   // "bottom" of butterfly pair

        Complex u = s_data[index1]; //even term
        Complex v = s_data[index2]; //odd term
        Complex v_tw = complex_mult(v, w);

        __syncthreads();
        s_data[index1] = complex_add(u, v_tw);
        s_data[index2] = complex_sub(u, v_tw);
        __syncthreads();
    }

    // 4) Output result to global memory
    d_stft_out[frame * N + t] = s_data[t];
}



void stft(const float* d_signal, const float* d_hann, Complex *data,int totalFrames,  int N,int H, int num_fft){

	int shared_bytes = N * sizeof(Complex);
	dim3 grid(num_fft);
	dim3 block(N);	
	stft_kernel_fixed<<<grid, block, shared_bytes>>>(d_signal, d_hann,data, totalFrames, N, H, num_fft);	
}

