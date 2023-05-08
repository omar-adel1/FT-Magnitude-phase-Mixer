import streamlit as st
from PIL import Image
# Set page configuration
st.set_page_config(page_title="FT-Magnitude-phase-Mixer", page_icon="✅", layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Add the content inside the container
st.markdown("<h3>Image 1</h3>", unsafe_allow_html=True)
static_picture, dynamic_pic, right_side = st.columns([2,2,4.5])
with static_picture:
    pic_1_upload = st.file_uploader("", type="jpg", accept_multiple_files= False)
with dynamic_pic:
    component_1 = st.selectbox(label="", options=[
    'FT Magnitude', 'FT Phase', 'FT Real component', 'FT Imaginary component'])
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
        image_choose_1 = st.selectbox(label="", options=['Image 1', 'Image 2'])
    with slider_1:
        selected_value_1 = st.slider("", 0, 100, step=1, format="%d%%")
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
        image_choose_2 = st.selectbox(label="", options=['Image 2', 'Image 1'])
    with slider_2:
        selected_value_2 = st.slider("", 0, 100, step=1, value = 5, format="%d%%")
    empty_2, mag_phase_2 = st.columns([1,3])
    with empty_2:
        st.write("")
    with mag_phase_2:
        mode_2 = st.selectbox(label="", options=['Phase','Magnitude','Imaginary','Real','Uniform magnitude','Uniform phase'])