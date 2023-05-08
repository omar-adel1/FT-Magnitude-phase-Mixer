import streamlit as st

# Create a container for the rows
container = st.container()

# Create the first row with three columns
with container:
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row1_col1.write("Column 1")
    row1_col2.write("Column 2")
    row1_col3.write("Column 3")

# Create the second row with two columns
with container:
    row2_col1, row2_col2 = st.columns(2)
    row2_col1.write("Column 1")
    row2_col2.write("Column 2")

# import streamlit as st
# from PIL import Image
# # Set page configuration
# st.set_page_config(page_title="FT-Magnitude-phase-Mixer", page_icon="✅", layout="wide")
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# # Add the content inside the container
# left_column,right_column = st.columns(2)
# with left_column:
#     with st.container():
#         st.markdown("<h3>Image 1</h3>", unsafe_allow_html=True)
#         static_picture,dynamic_pic = st.columns(2)
#         with static_picture:
#             pic_1_upload = st.file_uploader("", type="jpg")
#             if pic_1_upload is not None:
#                 image = Image.open(pic_1_upload)
#                 st.image(image)
#     with st.container():
#         st.markdown("<h3>Image 2</h3>", unsafe_allow_html=True)
#         st.write("This is the content inside the fieldset.")
#         st.write("You can add more content here.")

# with right_column:
#     st.write("fvververf")




# import streamlit as st
# from PIL import Image

# row1_container = st.container()
# with row1_container:
#     row1_col1, row1_col2 =  st.columns(2)
#     with row1_col1:
#         pic_1_upload = st.file_uploader("", type="jpg")
#         if pic_1_upload is not None:
#             image = Image.open(pic_1_upload)
#             st.image(image)

# row2_container = st.container()
# with row2_container:
#     row2_col1, row2_col2 =  st.columns(2)
#     row2_col1.write("Column 1")
#     row2_col2.write("Column 2")

