#include <cuda.h>
#include <cstdio>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "FFT_kernel.cu"


#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
void print_complex(const char* label, Complex c){
    printf("%s: %.5f + %.5fi\n",label,c.real,c.imag);
  }
int main(){
	
	const int N = 1024;
	const int num_ffts = 1;

	drwav wav;
    if (!drwav_init_file(&wav, "/scratch/apadi089/audio/fold1/101415-3-0-3.wav", NULL)) {
        printf("Failed to load file\n");
        return 1;
    }
	//audio file is read into a buffer, where each buffer[i] is an int16_t
    //int16_t* buffer = malloc(wav.totalPCMFrameCount * wav.channels * sizeof(int16_t));
    int16_t* buffer = new int16_t[wav.totalPCMFrameCount * wav.channels /** sizeof(int16_t)*/];
	drwav_read_pcm_frames_s16(&wav, wav.totalPCMFrameCount, buffer);
	
    printf("Read %llu frames from WAV file\n", wav.totalPCMFrameCount);

	//Fixme:	
	//float* mono = malloc(wav.totalPCMFrameCount * sizeof(float));
	float* mono = new float[wav.totalPCMFrameCount /* sizeof(float)*/];
	for(uint64_t i =0 ; i < wav.totalPCMFrameCount ; ++i){
		if(wav.channels == 1){	
		mono[i] = buffer[i]/32768.0f;
		}else{
		mono[i] = (buffer[i*2]+buffer[i*2+1])/(2.0f * 32768.0f);
		}	
}
	
	printf("First values of mono[]: ");
	for(unsigned int i = 0 ; i < 100; ++i){
		printf("Value at mono[%d]",i);		
		printf(": %f\n",mono[i]);	
	}
	
	//Should be changed to a power of 2
	//Complex* input = malloc(wav.totalPCMFrameCount * sizeof(Complex));
    Complex* input = new Complex[wav.totalPCMFrameCount /* sizeof(Complex)*/];

	for(unsigned int i = 0 ; i < wav.totalPCMFrameCount;++i){
			input[i].real = mono[i];
			input[i].imag = 0.0f;
		}
	
	printf("First values of input[]: ");
	for(unsigned int i = 0 ; i < 100; ++i){
		print_complex("value:",input[i]);
	}

	//Test Code Setup

	Complex* h_signal = new Complex[N * sizeof(Complex)];
	for (int i = 0; i < N; ++i) {
    	h_signal[i] = input[i];
	}
	
	
	Complex *d_signal; 
    cudaMalloc(&d_signal, N * sizeof(Complex)*num_ffts);
	cudaMemcpy(d_signal, h_signal, N * sizeof(Complex)*num_ffts, cudaMemcpyHostToDevice);
	fft(d_signal, N, num_ffts);
	size_t sharedBytes = N * sizeof(Complex);
//	fft(d_signal, N, num_ffts);
	cudaDeviceSynchronize();
	
	Complex h_result[N];
	cudaMemcpy(h_result, d_signal, N * sizeof(Complex)* num_ffts, cudaMemcpyDeviceToHost);

	printf("FFT result (N=%d):\n",N);
	for(int i = 0 ; i < N; ++i){
		printf("Value at %d",i);
		print_complex("value: ",h_result[i]);
	}

	cudaFree(d_signal);
	
	delete[] input;
	delete[] mono;
    delete[] buffer;
	return 0;

}
