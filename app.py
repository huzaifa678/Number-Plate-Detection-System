import io

import streamlit as st
import requests
import base64
from PIL import Image

URL = "http://localhost:8000/detect"

st.set_page_config(page_title="Object Detection App", page_icon="🔍", layout="centered")
st.title("🔍 AI Object Detection")
st.markdown("Upload an image")
st.sidebar.header("⚙️ Settings")

method = st.sidebar.multiselect(
    "Select OCR method",
    ["easy_ocr", "paddle"],
    default=["paddle"]
)

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.subheader("📷 Uploaded Image")
    st.image(image, use_container_width=True)
    st.divider()

    if st.button("Run Detection", use_container_width=True):

        with st.spinner("Running detection... please wait"):
            files = {
                "image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            data = {
                "methods": ",".join(method),
            }

            try:
                response = requests.post(URL, files=files, data=data)
                if response.status_code == 200:
                    result_data = response.json()
                    st.success("Detection Completed!")
                    col1, col2 = st.columns(2)
                    col1.metric("Texts found", len(result_data.get("results")))
                    st.write(result_data["results"])

                    if result_data.get("image"):
                        img_bytes = base64.b64decode(result_data["image"])
                        st.subheader("Detection Output")
                        st.image(img_bytes, use_container_width=True)
                    if result_data.get("cropped_images"):
                        st.subheader("Cropped Images")
                        for cropped_image in result_data["cropped_images"]:
                            img_bytes = base64.b64decode(cropped_image)

                            image = Image.open(io.BytesIO(img_bytes))

                            st.image(image, use_container_width=True)
                else:
                    st.error(f"Error: {response.status_code}")
                    st.code(response.text)

            except requests.exceptions.RequestException as e:
                st.error("Backend not reachable")
                st.exception(e)
else:
    st.info("👆 Upload an image to get started")


st.markdown("---")
st.caption("Built with ❤️ using Streamlit & FastAPI")
