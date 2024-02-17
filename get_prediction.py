import streamlit as st
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

st.write(f"## **TriVisioNet**: Get :blue[Age], :red[Gender] & :orange[Race]")


image_path = st.file_uploader(label="Upload an Image")

if image_path:
    image = np.expand_dims(img_to_array(
        load_img(image_path, target_size=(128, 128))), axis=0)
    st.image(image_path, width=300)


@st.cache_resource
def get_model():
    return load_model('./effnet1/kaggle/working/effnet')


model = get_model()

gender_list = ["Male", "Female"]
race_list = ["White", "Black", "Asian", "Indian", "Others"]

if st.button("Predict"):
    if image_path:
        pred = model.predict(image, verbose=0)
        pred_age = pred[0][0][0]
        pred_gender = gender_list[int(np.round(pred[1][0][0]))]
        pred_race = race_list[int(np.argmax(pred[2], axis=1))]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"### Age: :blue[{pred_age: .2f}]")
        with col2:
            st.write(f"### Gender: :red[{pred_gender}]")
        with col3:
            st.write(f"### Race: :orange[{pred_race}]")
    else:
        st.warning("Please upload an image")
