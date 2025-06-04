#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "complex.h"
#include "fft.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

int main() {
    drwav wav;
    if (!drwav_init_file(&wav, "/scratch/apadi089/audio/fold1/101415-3-0-3.wav", NULL)) {
        printf("Failed to load file\n");
        return 1;
    }
	//audio file is read into a buffer, where each buffer[i] is an int16_t
    int16_t* buffer = malloc(wav.totalPCMFrameCount * wav.channels * sizeof(int16_t));
    drwav_read_pcm_frames_s16(&wav, wav.totalPCMFrameCount, buffer);
	
    printf("Read %llu frames from WAV file\n", wav.totalPCMFrameCount);

	//Fixme:	
	float* mono = malloc(wav.totalPCMFrameCount * sizeof(float));
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
	Complex* input = malloc(wav.totalPCMFrameCount * sizeof(Complex));
    for(unsigned int i = 0 ; i < wav.totalPCMFrameCount;++i){
			input[i].real = mono[i];
			input[i].imag = 0.0f;
		}
	
	printf("First values of input[]: ");
	for(unsigned int i = 0 ; i < 100; ++i){
		print_complex("value:",input[i]);
	}
	//testing 
	//
	//
	Complex* test_signal = malloc(wav.totalPCMFrameCount * sizeof(Complex));
	for (int i = 0 ; i < 1024 ; ++i){
		test_signal[i] = input[i];
	}
	
	fast_fourier_transform(test_signal, wav.totalPCMFrameCount);
		
	printf("First values of input[]: ");
	for(unsigned int i = 0 ; i < 100; ++i){
		print_complex("value:",test_signal[i]);
	}
	free(buffer);
    drwav_uninit(&wav);
    return 0;
}

