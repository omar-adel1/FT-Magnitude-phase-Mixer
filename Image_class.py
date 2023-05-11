import cv2
import numpy as np 
import streamlit as st
import logging 

class Images:
    
    def __init__(self,image_path = None) :
        
        self.imagepath = image_path 
        if image_path is not None:
            
            self.image_read = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            self.image_read = cv2.resize(self.image_read,(190,190))
            self.img_shape = self.image_read.shape
            #Calculating Fourier transform of image
            self.ft = np.fft.fft2(self.image_read)
            self.fourier_shift = np.fft.fftshift(self.ft)
            self.magnitude = np.log(1+np.abs(self.fourier_shift))
            self.phase = np.angle(self.fourier_shift)
            self.real = self.fourier_shift.real
            self.imaginary = self.fourier_shift.imag
            self.uniform_magnitude = np.ones_like(self.magnitude)
            self.uniform_phase = np.zeros_like(self.phase)
        
    def get_component(self,component) :
        self.image_read = self.image_read / np.max(self.image_read)
        
        if component == "FT Magnitude" :
            return self.magnitude
        
        elif component == "FT Phase" :
            return self.phase
        
        elif component == "FT Real component" :
            return self.real
        
        elif component == "FT Imaginary component" :
            return self.imaginary    
        
        
            
    def display(self,component) :
        self.image_read = self.image_read / np.max(self.image_read)
        
        if component == "FT Magnitude" :
            st.image( self.magnitude)
        
        elif component == "FT Phase" :
            st.image( self.phase)
        
        elif component == "FT Real component" :
            st.image( self.real)
        
        elif component == "FT Imaginary component" :
            st.image( self.imaginary  )  


            
class ImageProcessor :
    
    def __init__(self) :
        self.image_1_component = None
        self.image_2_component = None
        self.Mixing_ratio = None
        self.mixing_output_1 = None
        self.mixing_output_2 = None
        self.image_1 = None
        self.image_2 = None
            

        
        
  
            
        
        
        
        