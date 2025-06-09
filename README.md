# CS147-FinalProject
Final Project For CS147 - GPU Programming

## Goal: Optimize Audio Data Pre-Proccessing for a CNN Classifier

## Introduction:

In order to be able to train a typical NN classifer to on audio, we need to both process this audio into a format that can be intepreted by a model, as well as proccess the audio into a form that carries with it information about the sample. This is where the Short Time Fourier Transform is perfect.

A typical Fourier Transform converts any signal from the time domain to the frequency domain, which then conveys how much of any given frequency is present in a given signal. While useful for simple signals, audio is very complex, in where frequencies can change throughout the lifetime of an audio sample. The STFT tells us both how much of a frequency is present in a signal, as well as **when** it is present. 

By performing a STFT on an audio signal, we obtain (usually) a 2D array containing the desired information. By taking the **Magnitude Squared**, we gain a spectrogram containing this information, which then can be passed into a NN as if it was an image.

In order to get our desired results, we must first **Window** our function, meaning we 'slice' the signal into desired frames. On each frame, we perform a **Fast Fourier Transform**, now given us both the how much and when we desire. Afterwards, we compute the spectorgram by taking the magnitude squared, and from that we achieve our proccessed audio. This must be peformed on EVERY audio sample, meaning with a GPU implementation, For a signal of size N and frame size of size M, we would need to atleast perform N/M FFT's, each which have a typical runtime of O(Nlog(n)). 

So the main steps are:
- Window
- Fast Fourier transform on num_frames
- Generate Spectrogram through magnitude squared

## Fast Fourier Transform: 

The fast fourier is a wonderful algorithm, as the inutuion behind it is simple(although everything else about it is not). For any signal X[n], we can seperate it into even and odd parts. It thene uses the idea of symetry for even an odd degree polynomials to compute a corresponding point at a given time, halving the amount of computation needed. Below is the serial imlementation I made for testing:
```
fast_fourier_transform(Complex* input, int N){
      //Root of unity 1
      if(N <= 1) return; 
          
      Complex* even = malloc(N/2 *sizeof(Complex));
      Complex* odd = malloc(N/2 *sizeof(Complex));
      for(int i = 0 ; i < N /2 ; ++i){
          even[i] = input[2*i];
          odd[i] = input[2*i+1];
      }
  
      fast_fourier_transform(even, N/2);
      fast_fourier_transform(odd, N/2);
      
       
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
```
And here is the corresponding output:

The issue however, is that this is a sequential, recursive algorithm. And this course isn't called "sequential programming". So I had to scrap this, and learn about the **2-Radix Butterfly Algorithm**. (Which hurt alot).

## cuFFT Implementation
Due to my human error, I also opted ot use my original main.cu that took care of the spectrogram and windowing, and alter it slighlty to instead use the cuFFT library. cuFFT essentially performs the same task, but with other optimizations to guratnee high performance and covering all edge cases. 

To use cuFFT, one must simply specify a 'plan', with the length, along with the number of frames (number of FFT's to perform). It then handles all bit reversal and indexing issues no matter the input. It executes all frames in parralel, having the desired functionality for our high demand computations. 

## Performance and Comparison

Both implementations, (my attempted cooley-Tuket and cuFFT), were used on 7 folds of the urbansound8k data set consisting of ~4 second long audio clips. The entire data set has around 16,000 samples, so without the exact number, each processed about 11,200 audio samples. The output where raw binary samples, as the NN classifer simply needs the values, and not a literal spectrogam to "look" at. 

### cuFFT: 

[Spectrogram of Dog Bark](spectrogram_101415-3-0-3-dog.png)
[Spectrogram of Dog Bark - 2](spectrogram_101415-3-0-8.png)
[Spectrogram of Gun Shot](cuFFT/spectrogram_102305-6-0-0-gun.png)
[Spectrogram of Jack Hammer](cuFFT/spectrogram_103074-7-0-0-jack.png)


## What I learned and Where to Improve

## Compile Commands
To compile the cooley-tukey version
```
nvcc -o ct_fft main.cu support.cu -lm
```
To compile cuFFT implementation
```
nvcc -I/usr/local/cuda-12.2/targets/x86_64-linux/include main.cu support.cu -o main -lcufft
```
To compile serial version
```
gcc -std=c99 -o main main.cu 
```
(mostly for my own documentation)

## Libraries and Resources: 

- dr_wav: A single file library, included as a header (used for loaded audio into buffer)
- Cuda C for GPU programming
- Numpy and Matplot for plotting spectrogram examples
- [Understanding Bit Reversal](https://youtu.be/gg2lgResMc0?si=rUICaErpVhQTzuQ0)
- [Understanding Butterfly's](https://youtu.be/EsJGuI7e_ZQ?si=I_uGoG0PrxT4MB_7)
- [Fast Fourier Transform Abstract](https://youtu.be/h7apO7q16V0?si=3SD3Lid2BQFgZxti)
