import cv2
import numpy as np 
import streamlit as st
import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -Logger name: %(name)s- Function: %(funcName)s - Line number : %(lineno)d - Level Name : %(levelname)s - massege : %(message)s ')
file_handler = logging.FileHandler('image_class.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
class Images:
    
    def __init__(self,image_path = None) :
        """
        Initializes an object of the class and sets the input image path if it is provided.
        If an image path is provided, the function reads the image from the provided path, 
        resizes it to 190x190 pixels using interpolation, and then calculates the Fourier 
        transform of the resized image. The Fourier transform components, i.e., magnitude, 
        phase, real, and imaginary parts of the Fourier transform are calculated and stored 
        as instance variables of the object. The magnitude is computed using a logarithmic 
        transformation to enhance visibility. The uniform magnitude and phase are set to ones 
        and zeros, respectively.
        """
        logger.info("______new rerun___")
        self.imagepath = image_path
        self.is_first_image = False
    
        if image_path is not None:
            # read image from the provided path and convert it to grayscale
            self.image_read = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            self.img_shape = self.image_read.shape
            
            # resize the image to 190x190 pixels using interpolation
            self.image_read = cv2.resize(self.image_read, (190, 190), interpolation=cv2.INTER_AREA)
            
            # calculate Fourier transform of the resized image
            self.ft = np.fft.fft2(self.image_read)
            self.fourier_shift = np.fft.fftshift(self.ft)
            
            # compute magnitude using logarithmic transformation
            self.magnitude = np.multiply(np.log10(1+np.abs(self.fourier_shift)), 20)
            
            # calculate phase, real, and imaginary parts of the Fourier transform
            self.phase = np.angle(self.fourier_shift)
            self.real = np.real(self.fourier_shift)
            self.imaginary = np.imag(self.fourier_shift)
            
            # set uniform magnitude to ones and uniform phase to zeros
            self.uniform_magnitude = np.ones_like(self.magnitude)
            self.uniform_phase = np.zeros_like(self.phase)
            logger.info("object image has been created") 
          
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
                logger.warning("Two images not same size")
                return False
            return True
      
    def get_component(self,component) :
        """
        Returns the specified Fourier component of the image.
        
        Args:
            component: a string indicating the Fourier component to retrieve. Possible values are:
                "Magnitude": the magnitude of the Fourier transform.
                "Phase": the phase of the Fourier transform.
                "Real": the real part of the Fourier transform.
                "Imaginary": the imaginary part of the Fourier transform.
                "Uniform magnitude": the Fourier transform with a uniform magnitude of 1.
                "Uniform phase": the Fourier transform with a uniform phase of 0.
        
        Returns:
            The specified Fourier component of the image.
        """
        if component == "Magnitude" :
            logger.info(' component {} has been called '.format(component))
            return np.abs(self.fourier_shift)        
        elif component == "Phase" :
            logger.info(' component {} has been called '.format(component))
            return self.phase
        elif component == "Real" :
            logger.info(' component {} has been called '.format(component))
            return self.real
        elif component == "Imaginary" :  
            logger.info(' component {} has been called '.format(component))
            return self.imaginary
        elif component == "Uniform magnitude" :
            logger.info(' component {} has been called '.format(component))
            return self.uniform_magnitude
        elif component == "Uniform phase" :
            logger.info(' component {} has been called '.format(component))
            return self.uniform_phase
            
    def display_component(self,component):
        """
        Displays the selected Fourier Transform (FT) component of an image.

        Args:
            component (str): The name of the component to display. Valid options are:
                - "FT Magnitude"
                - "FT Phase"
                - "FT Real component"
                - "FT Imaginary component"

        Returns:
            None
        """
        # Normalize the selected component using OpenCV's normalize function
        if component == "FT Magnitude":
            magnitude_normalized = cv2.normalize(self.magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            st.image(magnitude_normalized, clamp=True)
            logger.info(' component {} has been displayed '.format(component))
        elif component == "FT Phase":
            phase_normalized = cv2.normalize(self.phase, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            st.image(phase_normalized, clamp=True)
            logger.info(' component {} has been displayed '.format(component))
        elif component == "FT Real component":
            real_normalized = cv2.normalize(self.real, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            st.image(real_normalized, clamp=True)
            logger.info(' component {} has been displayed '.format(component))
        elif component == "FT Imaginary component":
            imaginary_normalized = cv2.normalize(self.imaginary, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            st.image(imaginary_normalized, clamp=True)
            logger.info(' component {} has been displayed '.format(component))

    
    def inverse_fourier_image(image):
        """
        This function performs the inverse Fourier transform on a given image using NumPy's ifft2 function,
        and returns the real part of the resulting array.
        """
        Inverse_fourier_image = np.real(np.fft.ifft2(image))  
        logger.info('inversefourier has been called')
        return Inverse_fourier_image
    

    @staticmethod
    def Mix_Images(img_1 :'Images' ,img_2 :'Images', component_image_1 : str,component_image_2 :str,Mix_ratio_1: float,Mix_ratio_2:float):
        """
        Mixes two images by combining their Fourier domain components based on user-specified ratios and components.
        
        Parameters:
        img_1 (Images): The first image object.
        img_2 (Images): The second image object.
        component_image_1 (str): The component of the first image to use in mixing.
        component_image_2 (str): The component of the second image to use in mixing.
        Mix_ratio_1 (float): The ratio of component_image_1 to mix.
        Mix_ratio_2 (float): The ratio of component_image_2 to mix.
        
        Returns:
        Mixed_img (np.ndarray): The mixed image as a numpy array.
        """

        #Get fourier parameters for each image 
        Mag_img1 = img_1.get_component("Magnitude")
        Phase_img1 = img_1.get_component("Phase")
        Real_img1 = img_1.get_component("Real")
        Imag_img1 = img_1.get_component("Imaginary")
        UniPhase_img1 = img_1.get_component("Uniform phase")
        UniMag_img1 = img_1.get_component("Uniform magnitude")
        
        Mag_img2 = img_2.get_component("Magnitude")
        Phase_img2 = img_2.get_component("Phase")
        Real_img2 = img_2.get_component("Real")
        Imag_img2 = img_2.get_component("Imaginary")
        UniPhase_img2 = img_2.get_component("Uniform phase")
        UniMag_img2 = img_2.get_component("Uniform magnitude")
        
        Mix_ratio_1 = Mix_ratio_1/100
        Mix_ratio_2 = Mix_ratio_2/100
        # Mix the components based on user-specified ratios and components
        if component_image_1 == "Magnitude"  and component_image_2 == "Phase":
            Mixed_Mag = Mag_img1*Mix_ratio_1 + Mag_img2*(1-Mix_ratio_1)
            Mixed_Phase = Phase_img2*Mix_ratio_2 + Phase_img1*(1-Mix_ratio_2)
            Mixed_FT = np.multiply(Mixed_Mag , np.exp(1j * Mixed_Phase ) )
            logger.info(' component1 : {} and component2: {}  has been selected '.format(component_image_1,component_image_2))
        elif component_image_1 == "Phase" and component_image_2 == "Magnitude" :
            Mixed_Phase = Phase_img1*Mix_ratio_1 + Phase_img2*(1-Mix_ratio_1)
            Mixed_Mag = Mag_img2*Mix_ratio_2 + Mag_img1*(1-Mix_ratio_2)
            Mixed_FT = np.multiply(Mixed_Mag , np.exp(1j * Mixed_Phase) )
            logger.info(' component1 : {} and component2: {}  has been selected '.format(component_image_1,component_image_2))
        elif component_image_1 == "Real" and component_image_2 == "Imaginary"   :
            New_real = Real_img1*Mix_ratio_1 + Real_img2*(1-Mix_ratio_1)
            New_Imag = Imag_img2*Mix_ratio_2 + Imag_img1*(1-Mix_ratio_2)
            Mixed_FT = New_real + 1j * New_Imag
            logger.info(' component1 : {} and component2: {}  has been selected '.format(component_image_1,component_image_2))
            
        elif component_image_1 == "Imaginary" and component_image_2 == "Real"  :
            New_Imag = Imag_img1*Mix_ratio_1 + Imag_img2*(1-Mix_ratio_1)
            New_real = Real_img2*Mix_ratio_2 + Real_img1*(1-Mix_ratio_2) 
            Mixed_FT = New_real + 1j * New_Imag
            logger.info(' component1 : {} and component2: {}  has been selected '.format(component_image_1,component_image_2))
            
        elif component_image_1 == "Magnitude" and component_image_2 == "Uniform phase" :
            Mixed_Mag = Mag_img1*Mix_ratio_1 + Mag_img2*(1-Mix_ratio_1)
            Mixed_Phase = UniPhase_img2*Mix_ratio_2 + Phase_img1*(1-Mix_ratio_2)   #Matrix of zeros 
            Mixed_FT = np.multiply(Mixed_Mag , np.exp(1j * Mixed_Phase))
            logger.info(' component1 : {} and component2: {}  has been selected '.format(component_image_1,component_image_2))
            
        elif component_image_1 == "Uniform phase" and component_image_2 == "Magnitude" :
            Mixed_Phase = UniPhase_img1*Mix_ratio_1 + Phase_img2*(1-Mix_ratio_1)
            Mixed_Mag = Mag_img2*Mix_ratio_2 + Mag_img1*(1-Mix_ratio_2)
            Mixed_FT = np.multiply(Mixed_Mag,np.exp(1j* Mixed_Phase))
            logger.info(' component1 : {} and component2: {}  has been selected '.format(component_image_1,component_image_2))
        
        elif component_image_1 == "Uniform magnitude"  and component_image_2 == "Phase":
            Mixed_Mag = UniMag_img1*Mix_ratio_1 + Mag_img2 * (1-Mix_ratio_1)   # Equivalent to UniMag_img1 or UniMag_img1
            Mixed_Phase = Phase_img2 * Mix_ratio_2 + Phase_img1*(1-Mix_ratio_2)
            Mixed_FT = np.multiply(Mixed_Mag,np.exp(1j * Mixed_Phase))
            logger.info(' component1 : {} and component2: {}  has been selected '.format(component_image_1,component_image_2))
        
        elif component_image_1 == "Phase" and component_image_2 == "Uniform magnitude" :
            Mixed_Phase = Phase_img1*Mix_ratio_1 + Phase_img2*(1-Mix_ratio_1)
            Mixed_Mag = UniMag_img2*Mix_ratio_2 + Mag_img1*(1-Mix_ratio_2)
            Mixed_FT = np.multiply(Mixed_Mag,np.exp(1j*Mixed_Phase))
            logger.info(' component1 : {} and component2: {}  has been selected '.format(component_image_1,component_image_2))
        
        elif component_image_1 == "Uniform magnitude" and component_image_2 == "Uniform phase":
            Mixed_Mag = UniMag_img1*Mix_ratio_1 + Mag_img2* (1-Mix_ratio_1)
            Mixed_Phase = UniPhase_img2*Mix_ratio_2 +Phase_img1*(1-Mix_ratio_2)
            Mixed_FT = np.multiply(Mixed_Mag, np.exp(1j * Mixed_Phase))
            logger.info(' component1 : {} and component2: {}  has been selected '.format(component_image_1,component_image_2))
            
        elif component_image_1 == "Uniform phase" and component_image_2 == "Uniform magnitude":
            Mixed_Phase = UniPhase_img1*Mix_ratio_1 + Phase_img2*(1-Mix_ratio_1)
            Mixed_Mag = UniMag_img2*Mix_ratio_2 + Mag_img1*(1-Mix_ratio_2)
            Mixed_FT =  np.multiply(Mixed_Mag, np.exp(1j * Mixed_Phase))  
            logger.info(' component1 : {} and component2: {}  has been selected '.format(component_image_1,component_image_2))
            
        else:
           st.warning("Invalid Combination")
           logger.warning(' Invalid Combination ')
           return
        Image_combined = np.real(np.fft.ifft2(np.fft.ifftshift(Mixed_FT)))
        Image_combined = cv2.normalize(Image_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        logger.info(' image combined succesfully ')
        # Image_combined = np.real(np.fft.ifft2(Mixed_FT))
        return Image_combined
    