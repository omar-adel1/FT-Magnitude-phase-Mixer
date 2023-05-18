import cv2
import numpy as np 
import streamlit as st
import logging 
logger = logging.getLogger(_name_)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -Logger name: %(name)s- Function: %(funcName)s - Line number : %(lineno)d - Level Name : %(levelname)s - massege : %(message)s ')
file_handler = logging.FileHandler('image_class.log',mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
class Images:
    
    def _init_(self,image_path = None):
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
        logger.info("___new rerun__")
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
      
    def get_component(self,component):
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
        if component == "Magnitude":
            logger.info(' component {} has been called '.format(component))
            return np.abs(self.fourier_shift)        
        elif component == "Phase":
            logger.info(' component {} has been called '.format(component))
            return self.phase
        elif component == "Real":
            logger.info(' component {} has been called '.format(component))
            return self.real
        elif component == "Imaginary":  
            logger.info(' component {} has been called '.format(component))
            return self.imaginary
        elif component == "Uniform magnitude":
            logger.info(' component {} has been called '.format(component))
            return self.uniform_magnitude
        elif component == "Uniform phase":
            logger.info(' component {} has been called '.format(component))
            return self.uniform_phase
            
    def display_component(self, component):
        """
        Retrieves the selected Fourier Transform (FT) component of an image.

        Args:
            component (str): The name of the component to retrieve. Valid options are:
                - "FT Magnitude"
                - "FT Phase"
                - "FT Real component"
                - "FT Imaginary component"

        Returns:
            np.ndarray: The normalized component image as a numpy array.
        """
        # Normalize the selected component using OpenCV's normalize function
        if component == "FT Magnitude":
            magnitude_normalized = cv2.normalize(self.magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            logger.info(' component {} has been retrieved '.format(component))
            return magnitude_normalized
        elif component == "FT Phase":
            phase_normalized = cv2.normalize(self.phase, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            logger.info(' component {} has been retrieved '.format(component))
            return phase_normalized
        elif component == "FT Real component":
            real_normalized = cv2.normalize(self.real, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            logger.info(' component {} has been retrieved '.format(component))
            return real_normalized
        elif component == "FT Imaginary component":
            imaginary_normalized = cv2.normalize(self.imaginary, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            logger.info(' component {} has been retrieved '.format(component))
            return imaginary_normalized


    @staticmethod
    def inverse_fourier_image(image):
        """
        This function performs the inverse Fourier transform on a given image using NumPy's ifft2 function,
        and returns the real part of the resulting array.
        """
        Inverse_fourier_image = np.real(np.fft.ifft2(np.fft.ifftshift(image))) 
        logger.info('inversefourier has been called')
        return Inverse_fourier_image
    
   # def set_option():
        
    
    # @staticmethod
    # def mix_components(component_1,component_2,Ratio):
    #     mixed_component = component_1*Ratio + component_2*(1-Ratio)
    #     return mixed_component
        

    @staticmethod
    def Mix_Images(img_1: 'Images', img_2: 'Images', component_image_1: str, component_image_2: str, Mix_ratio_1: float, Mix_ratio_2: float):
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

        # Get Fourier parameters for each image
        Fourier_components = {
            "Magnitude": [img_1.get_component("Magnitude"), img_2.get_component("Magnitude")],
            "Phase": [img_1.get_component("Phase"), img_2.get_component("Phase")],
            "Real": [img_1.get_component("Real"), img_2.get_component("Real")],
            "Imaginary": [img_1.get_component("Imaginary"), img_2.get_component("Imaginary")],
            "Uniform phase": [img_1.get_component("Uniform phase"), img_2.get_component("Uniform phase")],
            "Uniform magnitude": [img_1.get_component("Uniform magnitude"), img_2.get_component("Uniform magnitude")]
        }

        Mix_ratio_1 = Mix_ratio_1 / 100
        Mix_ratio_2 = Mix_ratio_2 / 100

        # Mix the components based on user-specified ratios and components
        if (component_image_1 in ["Real"] and component_image_2 in ["Imaginary"]) or  (component_image_1 in ["Imaginary"] and component_image_2 in ["Real"]):
            
            
            New_real = Fourier_components[component_image_1][0] * Mix_ratio_1 + Fourier_components[component_image_1][1] * (1 - Mix_ratio_1)
            New_Imag = Fourier_components[component_image_2][1] * Mix_ratio_2 + Fourier_components[component_image_2][0] * (1 - Mix_ratio_2)
            ratio_tuples = [New_real, New_Imag] if component_image_1 == "Real" else [New_Imag, New_real]
            Mixed_FT = ratio_tuples[0] + 1j * ratio_tuples[1]
            logger.info(f"Component 1: {component_image_1} and Component 2: {component_image_2} have been selected.")

     
        elif( component_image_1 in ["Magnitude", "Uniform magnitude"] and component_image_2 in ["Phase", "Uniform phase"]) or (component_image_1 in ["Phase", "Uniform phase"] and component_image_2 in ["Magnitude", "Uniform magnitude"]):
            ratio_tuples = ["Magnitude","Phase"] if component_image_1 in ["Magnitude", "Uniform magnitude"] else ["Phase","Magnitude"]
            Mixed_Mag = Fourier_components[component_image_1][0] * Mix_ratio_1 + Fourier_components[ratio_tuples[0]][1] * (1 - Mix_ratio_1)

            Mixed_Phase = Fourier_components[component_image_2][1] * Mix_ratio_2 + Fourier_components[ratio_tuples[1]][0] * (1 - Mix_ratio_2)
            ratio_tupless = [Mixed_Mag,Mixed_Phase] if component_image_1 in ["Magnitude", "Uniform magnitude"] else [Mixed_Phase,Mixed_Mag]
            Mixed_FT = np.multiply(ratio_tupless[0], np.exp(1j *ratio_tupless[1]))
            logger.info(f"Component 1: {component_image_1} and Component 2: {component_image_2} have been selected.")
        
    
        else:
            st.warning("Invalid Combination")
            logger.warning("Invalid Combination")
            return None

        Image_combined = Images.inverse_fourier_image(Mixed_FT)
        Image_combined = cv2.normalize(Image_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        logger.info("Image combined successfully")
        return Image_combined