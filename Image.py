import cv2
import numpy as np 
import streamlit as st
import logging 

class Image:
    
    def __init__(self,image_path) :
    
        self.image_read = cv2.imread(image_path,0)
        self.img_shape = self.image_read.shape
        #Calculating Fourier transform of image
        self.ft = np.fft.fft2(self.image_read)
        self.fourier_shift = np.fft.fftshift(self.ft)
        self.magnitude = np.abs(self.fourier_shift)
        self.phase = np.angle(self.fourier_shift)
        self.real = self.fourier_shift.real
        self.imaginary = self.fourier_shift.imag
        self.uniform_magnitude = np.ones_like(self.magnitude)
        self.uniform_phase = np.zeros_like(self.phase)
        
        