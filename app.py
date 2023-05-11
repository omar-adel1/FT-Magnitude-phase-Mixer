import streamlit as st
import logging as log
from PIL import Image
from Image_class import Images
# Set page configuration
st.set_page_config(page_title="FT-Magnitude-phase-Mixer", page_icon="✅", layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Add the content inside the container
# static_picture, dynamic_pic, right_side = st.columns([2,2,4.5])
left_side, right_side = st.columns([2,2.2])
with left_side:
    st.markdown("<h3>Image 1</h3>", unsafe_allow_html=True)
    static_picture, dynamic_pic = st.columns(2)
    with static_picture:   
        pic_1_upload = st.file_uploader("", type="jpg", accept_multiple_files= False)
        # Load the "waiting" image and resize it
        waiting_image_1 = Image.open("placeholder.png").resize((190, 190))  #placeholder 
        #waiting_image_1 = img(pic_1_upload) 
        # Display the resized image
        waiting_image_displayed_1 = st.image(waiting_image_1)
        if pic_1_upload is not None:
            # Use an empty string to delete the image
            waiting_image_displayed_1.empty()
            # Read the image from the file uploader
            image = Image.open(pic_1_upload)
            #image = Images(pic_1_upload)
            
           # image = img(pic_1_upload)
            # Resize the image to a width of 300 pixels
            resized_image_1 = image.resize((190, 190))
            #resized_image_1 = image.image_read
            # Display the resized image
            st.image(resized_image_1)
    with dynamic_pic:
        component_1 = st.selectbox(label="", options=[
        'FT Magnitude', 'FT Phase', 'FT Real component', 'FT Imaginary component'])
        # Load the "waiting" image and resize it
        waiting_image_2 = Image.open("placeholder.png").resize((190, 190))
        
        # Display the resized image
        st.image(waiting_image_2)
with right_side:
    mixer_outputs, selected_output=st.columns([1,2])
    with mixer_outputs:
        st.write("")
        st.markdown("<h4>Mixer output to:</h4>", unsafe_allow_html=True)
    with selected_output:
        mixer_components_1 = st.selectbox(label="", options=['Output 1', 'Output 2'])
    component__right_1, image_choose1, slider_1= st.columns([1,1,2])
    with component__right_1:
        st.write("")
        st.markdown("<h4>Component 1</h4>", unsafe_allow_html=True)
    with image_choose1:
        image_choose_1 = st.selectbox(label="",key="image_compo_1", options=['Image 1', 'Image 2'])
    with slider_1:
        selected_value_1 = st.slider("", 0, 100, step=1, value=50, key="slider_1", format="%d%%")
    empty_1, mag_phase_1 = st.columns([1,3])
    with empty_1:
        st.write("")
    with mag_phase_1:
        mode_1 = st.selectbox(label="", options=['Magnitude','Phase','Real','Imaginary','Uniform magnitude','Uniform phase'])
    component__right_2, image_choose2, slider_2= st.columns([1,1,2])
    with component__right_2:
        st.write("")
        st.markdown("<h4>Component 2</h4>", unsafe_allow_html=True)
    with image_choose2:
        image_choose_2 = st.selectbox(label="", key="image_compo_2", options=['Image 1', 'Image 2'])
    with slider_2:
        selected_value_2 = st.slider("", 0, 100, step=1, value = 50, key="slider_2", format="%d%%")
    empty_2, mag_phase_2 = st.columns([1,3])
    with empty_2:
        st.write("")
    with mag_phase_2:
        mode_2 = st.selectbox(label="",key="mode_2,phase", options=['Magnitude','Phase','Real','Imaginary','Uniform magnitude','Uniform phase'])
# static_picture_2, dynamic_pic_2, right_side_2 = st.columns([2,2,4.5])
left_side_2, right_side_2 = st.columns([2,2.2])
with left_side_2:
    st.markdown("<h3>Image 2</h3>", unsafe_allow_html=True)
    static_picture_2, dynamic_pic_2 = st.columns(2)
    with static_picture_2:
        pic_2_upload = st.file_uploader("", type="jpg",key="2_photo", accept_multiple_files= False)
        # Load the "waiting" image and resize it
        waiting_image_3 = Image.open("placeholder.png").resize((190, 190))

        # Display the resized image
        waiting_image_displayed_3 = st.image(waiting_image_3)
        if pic_2_upload is not None:
                # Remove the "waiting" image
            waiting_image_displayed_3.empty()
            # Read the image from the file uploader
            image = Image.open(pic_2_upload)

            # Resize the image to a width of 300 pixels
            resized_image_2 = image.resize((190, 190))

            # Display the resized image
            st.image(resized_image_2)
    with dynamic_pic_2:
        component_2 = st.selectbox(label="",key="component_2", options=[
        'FT Magnitude', 'FT Phase', 'FT Real component', 'FT Imaginary component'])
        # Load the "waiting" image and resize it
        waiting_image_4 = Image.open("placeholder.png").resize((190, 190))

        # Display the resized image
        st.image(waiting_image_4)
with right_side_2:
    output_1, output_2 = st.columns(2)
    with output_1:
        st.markdown("<h3>OutPut 1</h3>", unsafe_allow_html=True)
        waiting_image_5 = st.image("placeholder.png")
    with output_2:
        st.markdown("<h3>OutPut 2</h3>", unsafe_allow_html=True)
        waiting_image_6 = st.image("placeholder.png")