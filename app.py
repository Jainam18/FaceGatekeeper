import streamlit as st
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import tensorflow as tf
import os
from PIL import Image
# st.write("Hello World")
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('model.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""# Fake Face Detection""")

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)


def upload_predict(model,image):
    prediction = model.predict(image)
    preds = np.argmax(prediction)  
    return preds,prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image = image.resize((96, 96))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.reshape(image_array, (1, 96, 96, 3))
    preds, predictions = upload_predict(model,image_array)
    if preds==0:
        st.write(f"The Model has detected that it is a Fake Face with a probability score of {predictions[0][preds]}")
    else:
        st.write(f"The Model has detected that it is a Real Face with a probability score of {predictions[0][preds]}")