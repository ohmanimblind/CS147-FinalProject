#include <cuda.h>
#include <math.h>
#define PI  3.14159265358979323846f
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
  
  
  
 

__global__ void fftkernel(Complex *data, int N, int num_fft){

	int fft_index = blockIdx.x;
	int t = threadIdx.x;

	extern __shared__ Complex s_data[];

	int data_id = fft_index * N + t;

	s_data[t] = data[data_id];
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
	data[data_id] = s_data[t];
	
}

void fft(Complex *data, int N, int num_fft){

	int shared_bytes = N * sizeof(Complex);
	
	fftkernel<<<num_fft, N, shared_bytes>>>(data, N, num_fft);	
}

