#include "complex.h"
#include <stdlib.h>
#include <stdio.h>
int main(){
Complex test1 = {2.0f,3.0f};
   Complex test2= {1.0f,-4.0f};
  
      //complex_sub(test1,test2);
   Complex mult_test = complex_mult(test1,test2);
  print_complex("test1: ", test1);	
  print_complex("test2: ", test2);		
  print_complex("mult_test: ", mult_test);	

return 0;
}
