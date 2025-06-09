#ifndef COMPLEX_H_
#define COMPLEX_H_
#include <float.h>
#include <stdio.h>



typedef struct{
	float real;
	float imag;	
}Complex;

Complex complex_add(Complex a, Complex b){
		return (Complex){a.real + b.real, a.imag + b.imag};
}
Complex complex_sub(Complex a, Complex b){
		return(Complex){a.real - b.real, a.imag - b.imag};
}

Complex complex_mult(Complex a, Complex b){
	   return(Complex){a.real*b.real-a.imag*b.imag, a.imag*b.real+b.imag*a.real};
}
	


void print_complex(const char* label, Complex c){
	printf("%s: %.5f + %.5fi\n",label,c.real,c.imag);
}
#endif
