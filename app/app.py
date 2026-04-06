import streamlit as st
import numpy as np
from PIL import Image
import io

from model import predict
from utils import preprocess_image, postprocess_image

st.set_page_config(
    page_title="Image Colorization",
    layout="wide"
)

st.markdown("""
<style>

/* Title */
h1 {
    text-align: center;
}

/* Subheaders */
h3 {
    color: #A6A6A6;
}

/* Card-like containers */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Buttons */
.stDownloadButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    font-size: 16px;
}

/* Upload box */
.stFileUploader {
    border: 2px dashed #555;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)




# Title
# st.title("Image Colorization")
# st.write("Upload a grayscale and see it colorized using deep learning.")
st.markdown("""
<h1>AI Image Colorization</h1>
<p style='text-align:center; color:gray;'>
Upload a grayscale and see it colorized using deep learning.
</p>
""", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### Original")
        st.image(image, width=400)

    # Process
    L, meta = preprocess_image(image)
    pred_ab = predict(L)
    output = postprocess_image(L, pred_ab, meta)

    with col2:
        st.markdown("### Colorized")
        st.image(output, width=400)


    # Download Button
    st.subheader("Download Result")

    # Convert to bytes
    result_image = Image.fromarray(output)
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="download",
        data=byte_im,
        file_name="colorized.png",
        mime="image/png"
    )