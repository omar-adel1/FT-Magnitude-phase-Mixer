import cv2
import numpy as np 
import streamlit as st
import logging 

class Images:
    
    def __init__(self,image_path = None) :
        
        self.imagepath = image_path 
        self.is_first_image = False
        if image_path is not None:
            
            self.image_read = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8),cv2.IMREAD_GRAYSCALE)
            self.img_shape = self.image_read.shape
            
            self.image_read = cv2.resize(self.image_read, (190, 190), interpolation=cv2.INTER_AREA)
            
              #Calculating Fourier transform of image 
            #All components are 2d matrices of the size which is resized (190,190)
            self.ft = np.fft.fft2(self.image_read)
            self.fourier_shift = np.fft.fftshift(self.ft)
            self.magnitude =np.multiply( np.log10(1+np.abs(self.fourier_shift)),20)  #Logarithmic Transformation
           # self.magnitude = np.abs(self.fourier_shift)
            self.phase = np.angle(self.fourier_shift)
            self.real = np.real(self.fourier_shift)
            self.imaginary = np.imag(self.fourier_shift)
            self.uniform_magnitude = np.ones_like(self.magnitude)
            self.uniform_phase = np.zeros_like(self.phase)
             
          
    """
    Function which checks size of the two images
     Check the size of the second image
     check_size(img1, image_path=file2)          
    """ 
    
    def set_first_image(self):
        self.is_first_image = True
             
    def set_second_image(self):
        self.is_first_image = False
    
        
    @staticmethod
    def check_size(image_1,image_2) :
        if image_1 and image_2 :
            
            if image_1.img_shape != image_2.img_shape:
                st.warning("Two images not same size")
                return False
            return True
        
    """
    Function get_component :
    Input : component : string
    
    returns----> normalized 2d matrix representing one of the components
    
    """  
      
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
     """   
        
        elif component == "Uniform phase" :
            return self.uniform_phase
            
    def display_component(self,component) :
       # self.image_read = self.image_read / np.max(self.image_read)
        
        
        if component == "FT Magnitude" :
            magnitude_normalized = cv2.normalize(self.magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            st.image(magnitude_normalized, clamp=True)
            #st.image( self.magnitude,clamp=True)
        
        elif component == "FT Phase" :
            phase_normalized = cv2.normalize(self.phase, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            st.image(phase_normalized, clamp=True)
           # st.image( self.phase,clamp=True)
        
        elif component == "FT Real component" :
            
            real_normalized = cv2.normalize(self.real, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            st.image(real_normalized, clamp=True)
           # st.image( self.real,clamp=True)
        
        elif component == "FT Imaginary component" :
            
             imaginary_normalized = cv2.normalize(self.imaginary, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
             st.image(imaginary_normalized, clamp=True)
          #  st.image( self.imaginary , clamp=True )  




"""        
class ImageProcessor :
    
    def __init__(self,image1,image2) :
        self.image_1_component = None
        self.image_2_component = None
        self.Mixing_ratio = None
        self.mixing_output_1 = None
        self.mixing_output_2 = None
        self.image_1 = None
        self.image_2 = None
            

"""        
        
  
            
        
        elif component_image_1 == "Uniform magnitude"  and component_image_2 == "Phase":
            Mixed_Mag = UniMag_img1*Mix_ratio_1 + Mag_img2 * (1-Mix_ratio_1)   # Equivalent to UniMag_img1 or UniMag_img1
            Mixed_Phase = Phase_img2 * Mix_ratio_2 + Phase_img1*(1-Mix_ratio_2)
            Mixed_FT = np.multiply(Mixed_Mag,np.exp(1j * Mixed_Phase))
        
        elif component_image_1 == "Phase" and component_image_2 == "Uniform magnitude" :
            Mixed_Phase = Phase_img1*Mix_ratio_1 + Phase_img2*(1-Mix_ratio_1)
            Mixed_Mag = UniMag_img2*Mix_ratio_2 + Mag_img1*(1-Mix_ratio_2)
            Mixed_FT = np.multiply(Mixed_Mag,np.exp(1j*Mixed_Phase))
        
        