import cv2
import numpy as np 
import streamlit as st
import streamlit as st
import logging as log
from PIL import Image
from Image_class import Images
def main():
    st.set_page_config(page_title="Image Fourier Transform", page_icon=":guardsman:", layout="wide")
    st.title("Image Fourier Transform")
    st.write("Upload an image and view its Fourier transform components")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jfif","png"], accept_multiple_files=False)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Images(uploaded_file)
        st.write("Original Image")
        st.image(image.image_read)
        
        # Fourier Transform components
        st.write("Fourier Transform Components")
        component = st.selectbox("Select a Fourier Transform component to view", 
                                ("FT Magnitude", "FT Phase", "FT Real component", "FT Imaginary component"))
        
        image.display_component(component)
        
if __name__ == "__main__":
    main()
