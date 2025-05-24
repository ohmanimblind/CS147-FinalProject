#ifndef FFT_H_
#define FFT_H_

#include "complex.h"
#include <math.h>
#define PI 3.14159265358979

//Input is array of complex values coming from .wav
fast_fourier_transform(Complex* input, int N){
	//Root of unity 1
	if(N <= 1) return; 
		
	Complex* even = malloc(N/2 *sizeof(Complex));
	Complex* odd = malloc(N/2 *sizeof(Complex));
	for(int i = 0 ; i < N /2 ; ++i){
		even[i] = input[2*i];
		odd[i] = input[2*i+1];
	}
	//define w 
	fast_fourier_transform(even, N/2);
	fast_fourier_transform(odd, N/2);
	
	//FIX ME 
	for (unsigned int j = 0; j < N /2 ; ++j){
		float exponent = -2 * PI * j / N;
		Complex w = {cosf(exponent),sinf(exponent)};
		Complex rhs = complex_mult(w, odd[j]);		

		input[j] = complex_add(even[j],rhs);
		input[j + N/2] = complex_sub(even[j],rhs);

	}



	

	free(even);
	free(odd);

}


#endif


