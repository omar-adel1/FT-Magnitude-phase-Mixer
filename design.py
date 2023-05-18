    
import streamlit as st
import logging 
from PIL import Image
from Image_class import Images
# Set page configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -Logger name: %(name)s- Function: %(funcName)s - Line number : %(lineno)d - Level Name : %(levelname)s - massege : %(message)s ')
file_handler = logging.FileHandler('design.log',mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
def main():
    # initialize session state
    if "output_1_image" not in st.session_state:
        st.session_state.output_1_image = None
    if "output_2_image" not in st.session_state:
        st.session_state.output_2_image = None
    logger.warning("__________Streamlit rerun__________")
    st.set_page_config(page_title="FT-Magnitude-phase-Mixer", page_icon="🎨", layout="wide")
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    # Add the content inside the container
    # static_picture, dynamic_pic, right_side = st.columns([2,2,4.5])
    left_side, right_side = st.columns([2,2.2])
    with left_side:
        st.markdown("<h3>Image 1</h3>", unsafe_allow_html=True)
        static_picture, dynamic_pic = st.columns(2)
        with static_picture:   
            pic_1_upload = st.file_uploader("", type=["jpg", "jfif","png"], accept_multiple_files=False)
            # Load the "waiting" image and resize it
            waiting_image_1 = Image.open("placeholder.png").resize((190, 190))  #placeholder 
            #waiting_image_1 = image(pic_1_upload) 
            # Display the resized image
            waiting_image_displayed_1 = st.image(waiting_image_1)
            if pic_1_upload is not None:
                # Use an empty string to delete the image
                logger.info("Picture 1 Uploaded")
                waiting_image_displayed_1.empty()
                # Read the image from the file uploader
                # image = Image.open(pic_1_upload)
                image_1 = Images(pic_1_upload)
                st.image(image_1.image_read)
                
                # resized_image_1 = Image.fromarray(image.image_read)
                # resized_image_1 = image.resize((190, 190))

                # st.image(resized_image_1)
        with dynamic_pic:
            component_1 = st.selectbox(label="", options=[
            'FT Magnitude', 'FT Phase', 'FT Real component', 'FT Imaginary component'])
            # Load the "waiting" image and resize it
            waiting_image_2 = Image.open("placeholder.png").resize((190, 190))
            # if image.imagepath is not None:
            #     component = image.get_component(component_1)
            #     component_image = Image.fromarray(component)
            # else:
            #     component_image = waiting_image_2
            # #    waiting_image_2 = image.get_component(component_1)
            # # Display the resized image
            waiting_image_displayed_2 = st.image(waiting_image_2)
            logger.info("Picture 1 Component selected is:{}".format(component_1))
            if pic_1_upload is not None:
                waiting_image_displayed_2.empty()
                image_component_1 = image_1.display_component(component_1)
                st.image(image_component_1, clamp = True)
                logger.info("Image 1 component displayed")
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
            image_choose_2 = st.selectbox(label="", key="image_compo_2", options=['Image 2', 'Image 1'])
        with slider_2:
            selected_value_2 = st.slider("", 0, 100, step=1, value = 50, key="slider_2", format="%d%%")
        empty_2, mag_phase_2 = st.columns([1,3])
        with empty_2:
            st.write("")
        with mag_phase_2:
            mode_2 = st.selectbox(label="",key="mode_2,phase", options=['Phase','Magnitude','Real','Imaginary','Uniform magnitude','Uniform phase'])
    # static_picture_2, dynamic_pic_2, right_side_2 = st.columns([2,2,4.5])
    left_side_2, right_side_2 = st.columns([2,2.2])
    with left_side_2:
        st.markdown("<h3>Image 2</h3>", unsafe_allow_html=True)
        static_picture_2, dynamic_pic_2 = st.columns(2)
        with static_picture_2:
            pic_2_upload = st.file_uploader("", type=["jpg", "jfif","png"],key="2_photo", accept_multiple_files= False)
            # Load the "waiting" image and resize it
            waiting_image_3 = Image.open("placeholder.png").resize((190, 190))

            # Display the resized image
            waiting_image_displayed_3 = st.image(waiting_image_3)
            if pic_2_upload is not None:
                    # Remove the "waiting" image
                waiting_image_displayed_3.empty()
                image_2 = Images(pic_2_upload)
                if pic_1_upload is not None:
                    if image_1.img_shape != image_2.img_shape:
                        logger.info("Picture is not the same size")
                        st.warning("Not the same size")
                        return
                logger.info("Picture 2 Uploaded")
                st.image(image_2.image_read)
        with dynamic_pic_2:
            component_2 = st.selectbox(label="",key="component_2", options=[
            'FT Magnitude', 'FT Phase', 'FT Real component', 'FT Imaginary component'])
            # Load the "waiting" image and resize it
            waiting_image_4 = Image.open("placeholder.png").resize((190, 190))
            # Display the resized image
            waiting_image_displayed_4 = st.image(waiting_image_4)
            logger.info("Picture 2 Component selected is:{}".format(component_1))
            if pic_2_upload is not None:
                waiting_image_displayed_4.empty()
                image_component_2 = image_2.display_component(component_2)
                st.image(image_component_2, clamp = True)
                logger.info("Image 2 component displayed")
    with right_side_2:
        output_1, output_2 = st.columns(2)
        with output_1:
            st.markdown("<h3>Output 1</h3>", unsafe_allow_html=True)
            waiting_image_5 = Image.open("placeholder.png").resize((190, 190))
            waiting_image_displayed_5 = st.image(waiting_image_5)
            if mixer_components_1 == "Output 1" and pic_2_upload is not None and pic_1_upload is not None:
                waiting_image_displayed_5.empty()
                logger.info("Mixer of component 1 and component 2 being displayed in Output 1")
                if image_choose_1 != image_choose_2:
                    st.session_state.output_1_image = Images.Mix_Images(image_1, image_2, mode_1, mode_2, selected_value_1, selected_value_2)
                elif image_choose_1 == image_choose_2 == "Image 1":
                    st.session_state.output_1_image = Images.Mix_Images(image_1, image_1, mode_1, mode_2, selected_value_1, selected_value_2)
                elif image_choose_1 == image_choose_2 == "Image 2":
                    st.session_state.output_1_image = Images.Mix_Images(image_2, image_2, mode_1, mode_2, selected_value_1, selected_value_2)
            if st.session_state.output_1_image is not None:
                waiting_image_displayed_5.empty()
                st.image(st.session_state.output_1_image, clamp= True)
            logger.info("Component 1 image : {} , Component 2 image: {} \n Component 1 mode : {} , Component 2 mode: {} \n Component 1 Slider value : {} , Component 2 Slider value: {}".format(image_choose_1, image_choose_2, mode_1, mode_2, selected_value_1, selected_value_2))
        with output_2:
            st.markdown("<h3>Output 2</h3>", unsafe_allow_html=True)
            waiting_image_6 = Image.open("placeholder.png").resize((190, 190))
            waiting_image_displayed_6 = st.image(waiting_image_6)
            if mixer_components_1 == "Output 2" and pic_2_upload is not None and pic_1_upload is not None:
                waiting_image_displayed_6.empty()
                logger.info("Mixer of component 1 and component 2 being displayed in Output 2")
                if image_choose_1 != image_choose_2:
                    st.session_state.output_2_image = Images.Mix_Images(image_1, image_2, mode_1, mode_2, selected_value_1, selected_value_2)
                elif image_choose_1 == image_choose_2 == "Image 1":
                    st.session_state.output_2_image = Images.Mix_Images(image_1, image_1, mode_1, mode_2, selected_value_1, selected_value_2)
                elif image_choose_1 == image_choose_2 == "Image 2":
                    st.session_state.output_2_image = Images.Mix_Images(image_2, image_2, mode_1, mode_2, selected_value_1, selected_value_2)
            if st.session_state.output_2_image is not None:
                waiting_image_displayed_6.empty()
                st.image(st.session_state.output_2_image, clamp= True)
            logger.info("Component 1 image : {} , Component 2 image: {} \n Component 1 mode : {} , Component 2 mode: {} \n Component 1 Slider value : {} , Component 2 Slider value: {}".format(image_choose_1, image_choose_2, mode_1, mode_2, selected_value_1, selected_value_2))