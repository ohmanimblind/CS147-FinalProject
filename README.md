# CS147-FinalProject
Final Project For CS147 - GPU Programming

## Goal: Optimize Audio Classification with Neural Networks through GPU Implementation

## Project Flow:

For each audio data point, we need to perform:
STFT (achieving a 2D array) -> STFT^2 (To achieve Spectrogram) 

Afterwards, they could be passed into the NN for classification. 

There are oppurtunities to optimize in the STFT aspects (windows can be performed in parallel) and self multiplication of STFT (essentially speeding up matrix multiply). There are based methods to improve NN performance with a GPU implementation. 

## Libraries: 

- Librosa for Audio Proccessing
- Numba + Python for GPU Aspect
- TensorFlow for NN creation
