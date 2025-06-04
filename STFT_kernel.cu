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

__global__ void stftkernel(const float* d_signal, const float* d_hann, Complex *data,int totalFrames, int N,int H, int num_fft){

	int fft_index = blockIdx.x;
	int t = threadIdx.x;

	extern __shared__ Complex s_data[];
	
	int data_id = fft_index * N + t;

	//make partitions and apply hann

	int start = fft_index * H;
	int end = start + t;
	float windowed = 0.0f;
	
	if(end < totalFrames){
		windowed = d_signal[end] * d_hann[t];
	}
	s_data[t].real = windowed;
	s_data[t].imag = 0.0f;


	__syncthreads();

   //Bit magic for log(N)
   int logN = __ffs(N) - 1;
   //^ insane

   for(int s = 1; s <= logN; s++){
		int m = 1 << s; //for powers of 2
		int halfM = m >> 1; //division by 2
	
    	int k = t & (halfM - 1);
    	float angle = -2.0f * PI * k / float(m);
    	Complex w = {cosf(angle),sinf(angle)};  
	
		int groupStart = (t >> s) * m ;
    	int index1 = groupStart +k;
    	int index2 = index1 + halfM;
		Complex c1 = s_data[index1];
		Complex c2 = s_data[index2];
 
	    Complex c3 = complex_mult(c2,w);
    	__syncthreads();

		s_data[index1] = complex_add(c1, c3);
		s_data[index2] = complex_sub(c1,c3);
		__syncthreads();
		
	}
	//data[data_id] = s_data[t];
	data[fft_index * N + t] = s_data[t];
}

void stft(const float* d_signal, const float* d_hann, Complex *data,int totalFrames,  int N,int H, int num_fft){

	int shared_bytes = N * sizeof(Complex);
	dim3 grid(num_fft);
	dim3 block(N);	
	stftkernel<<<grid, block, shared_bytes>>>(d_signal, d_hann,data, totalFrames, N, H, num_fft);	
}

