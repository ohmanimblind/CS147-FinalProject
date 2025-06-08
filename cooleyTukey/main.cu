#include <cuda.h>
#include <cstdio>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "support.h"
#include "STFT_kernel.cu"


#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
void print_complex(const char* label, Complex c){
    printf("%s: %.5f + %.5fi\n",label,c.real,c.imag);
  }
int main(int argc, char *argv[]){
	Timer timer;	
	/*PART ONE: AUDIO PROCESSING */
	/*----------------------------
	Preprocessing Audio using the dr_wav single file header. 
	Much of the processing I am not familar with, so most of it is simply magic
	Steps are as follows:
		- use dr_wav to load audio file into buffer
		- process the audio through channel magic and produce a 'mono' array

	*/
	char* wav_path = argv[1];
	drwav wav;
    if (!drwav_init_file(&wav, wav_path, NULL)) {
        printf("Failed to load file\n");
        return 1;
    }
	uint32_t fs = wav.sampleRate;
   printf("Sample rate: %u Hz\n",fs);
	//audio file is read into a buffer, where each buffer[i] is an int16_t
    //int16_t* buffer = malloc(wav.totalPCMFrameCount * wav.channels * sizeof(int16_t));
    int16_t* buffer = new int16_t[wav.totalPCMFrameCount * wav.channels /** sizeof(int16_t)*/];
	drwav_read_pcm_frames_s16(&wav, wav.totalPCMFrameCount, buffer);
	
    //printf("Read %llu frames from WAV file\n", wav.totalPCMFrameCount);

   startTime(&timer);	
	//Audio nonsense I dont understand: Has to do with channels
	float* h_mono = new float[wav.totalPCMFrameCount /* sizeof(float)*/];
	for(uint64_t i =0 ; i < wav.totalPCMFrameCount ; ++i){
		if(wav.channels == 1){	
		h_mono[i] = buffer[i]/32768.0f;
		}else{
		h_mono[i] = (buffer[i*2]+buffer[i*2+1])/(2.0f * 32768.0f);
		}	
    }	
	stopTime(&timer);
/*	printf("First values of h_mono[]: ");
	for(unsigned int i = 0 ; i < 10; ++i){
		printf("Value at h_mono[%d]",i);		
		printf(": %f\n",h_mono[i]);	
	}*/

	/*PART TWO: SETTING STFT KERNEL PARAMETERS */
	/*----------------------------
	The meat of the project. The needed parameters of the STFT are as follows:
	N  := Size of the window that a FFT is performed on
	H := Hop size, how much we "shift" over before performing another sample
	h_hann := The windowing function that 'chunks' our signal. I used a standard window function
	num_frames := how many frames we will have, and totalFrames := lenght of the signal
	------------------------------
	*/

	startTime(&timer);
	//STFT SECTION:
	const int N = 1024; //window size
    const int H = 512;	//hop size
	int totalFrames = (int)wav.totalPCMFrameCount;	

	//frames
	int num_frames = (totalFrames - N)/ H + 1;

	//declare standard Hann window
	float* h_hann = new float[N]; //size of our window
	
	for (int i = 0; i < N; ++i){
			h_hann[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (float)(N-1)));
	}

	//Allocations:
	float* d_input;
	float* d_hann;
	Complex* d_stft;

	cudaMalloc((void**)&d_input, totalFrames * sizeof(float));
	cudaMemcpy(d_input, h_mono, totalFrames * sizeof(float), cudaMemcpyHostToDevice);
  
	cudaMalloc((void**) &d_hann, N* sizeof(float));    
	cudaMemcpy(d_hann, h_hann, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_stft, (size_t)num_frames * N * sizeof(Complex));
	
	stft(d_input, d_hann, d_stft, totalFrames, N, H, num_frames);
	
	cudaDeviceSynchronize();
 	
	//Complex* stft_result = (Complex*)malloc((size_t)num_frames * N * sizeof(Complex));
	//cudaMemcpy(stft_result, d_stft, (size_t)num_frames * N * sizeof(Complex), cudaMemcpyDeviceToHost);
	
	/*for(int j = 0; j < 10 ;++j){
		print_complex("stft value:",stft_result[j]);	
	}*/
	
	stopTime(&timer);
	//printf("Time for stft: %.5d\n",elapsedTime(timer));

	/*PART THREE: SPECTROGRAM */
	/*----------------------------
	One of the last steps of the pipeline, we invoke a simple kernel to compute 
	the magnitude squared of our STFT_output. This gives us a 2D array (but organized in a 1D manner)
	of real values, that we can used to produce a heatmap, or a spectogram. This is what we feed into a 
	CNN for classification. Instead of matrix mult., we simply square the real and imaginary parts of our
	fourier coeffecients, and add them
	------------------------------
	NOTE: possible improvement here by keeping data on GPU before returning	
	*/

	float* spec_output = new float[(size_t)num_frames * N ];
	float* d_spec;
	//Complex *d_stftin;
	
	//cudaMalloc((void**)&d_stftin, (size_t)num_frames * N * sizeof(Complex));
	//cudaMemcpy(d_stftin, stft_result, (size_t)num_frames* N * sizeof(Complex), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_spec, (size_t)num_frames * N * sizeof(float));
		

	make_spectro(d_stft, d_spec ,num_frames, N );
	cudaDeviceSynchronize();	
	cudaMemcpy(spec_output, d_spec, (size_t)num_frames * N * sizeof(float), cudaMemcpyDeviceToHost);

	/*	for(int j = 0; j < 10 ;++j){
		printf("spec value: %.8f\n",spec_output[j]);	
	}*/

	//TESTETST TEST OUTPUT
	FILE* f = fopen("spectrogram_ct.bin","wb");
	if(f){
		fwrite(spec_output,sizeof(float),(size_t)num_frames*N,f);
		fclose(f);
	}else{
		printf("uh oh");	
	}
	
	printf("numFrames: %d\n",num_frames);
	cudaFree(d_input);
	cudaFree(d_hann);
	cudaFree(d_stft);

	cudaFree(d_spec);
	delete[] spec_output;
	delete[] h_hann;
	delete[] h_mono;
	delete[] buffer;
	drwav_uninit(&wav);

	return 0;
}
