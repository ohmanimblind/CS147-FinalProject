#include <cuda.h>
#include <math.h>
#define PI  3.14159265358979323846f
#define BLOCK_SIZE 256
	//<<< >>>;
	//fftkernel<<< >>>;
/*
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
  } */
  
  
 __global__ void spectrogram(const cufftComplex* d_stft, float* d_spectro, int totalBins){
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if(t >= totalBins) return;	

	float re = d_stft[t].x;
	float im = d_stft[t].y;

	d_spectro[t] = re * re + im * im;

} 

void make_spectro(const cufftComplex* d_stft, float* d_spectro, int num_frames, int N){
	int total_Bins = num_frames * (N/2);
	dim3 block_size(BLOCK_SIZE);
	dim3 grid = (total_Bins + BLOCK_SIZE - 1) / BLOCK_SIZE;
	spectrogram<<<grid, BLOCK_SIZE >>>(d_stft,d_spectro, total_Bins);
} 

__global__ void complexify(const float* d_windowed, cufftComplex* d_complexIn ,int size){

	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if(t >= size) return;
		d_complexIn[t].x = d_windowed[t];
		d_complexIn[t].y = 0.0f;		
}

void call_complexify(const float* d_windowed, cufftComplex* d_complexIn ,int size){

dim3 grid((size + BLOCK_SIZE-1)/BLOCK_SIZE);	

	complexify<<<grid,BLOCK_SIZE>>>(d_windowed, d_complexIn, size);
}


extern "C"
__global__ void stft_kernel_fixed(
    const float* d_signal,
    const float* d_hann,
    float*      d_windowed,
    int           totalSamples,
    int           N,
    int           H,
    int           num_frames
) {
    int frame = blockIdx.x;
    int t     = threadIdx.x;
    if (frame >= num_frames || t >= N) return;


    // 1) load & window
    int start = frame * H;
    int idx   = start + t;
    float windowed = 0.0f;
    if (idx < totalSamples) {
       windowed = d_signal[idx] * d_hann[t];
    }
    __syncthreads();

  

    // 4) Output result to global memory
    d_windowed[frame * N + t] = windowed;
}



void stft(const float* d_signal, const float* d_hann, float* data,int totalFrames,  int N,int H, int num_fft){

	//int shared_bytes = N * sizeof(Complex);
	dim3 grid(num_fft);
	dim3 block(N);	
	stft_kernel_fixed<<<grid, block /*,shared_bytes*/>>>(d_signal, d_hann,data, totalFrames, N, H, num_fft);	
}

