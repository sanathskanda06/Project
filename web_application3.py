import streamlit as st
import numpy as np
import keras.utils as image
import tensorflow as tf


# Load your trained model
model = tf.keras.models.load_model('C:/Users/sanat/Downloads/model2(1).hdf5')

st.title('Alzheimers Classification')

images =  st.file_uploader("Choose an image...", type=("jpeg","img","jpg"))


if images is not None:
    
    img = image.load_img(images, target_size=(224,224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    




# Make a prediction on the image
    prediction = model.predict(img_tensor)

    # Get the class with the highest probability
     
    prediction = np.argmax(prediction)
    st.image(images, caption="Uploaded image", use_column_width=False)
    st.title("The result is:")
    st.title(prediction)
   
