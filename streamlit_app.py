import streamlit as st
import numpy as np
from keras.preprocessing import image
import base64
import io

# Load the Keras model
from keras.models import load_model
model = load_model('best_model.h5')  # Replace 'your_model.h5' with the actual filename of your model

st.title("Image Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
img_data = None

if uploaded_file is not None:
    # Use in-memory image
    image_stream = io.BytesIO(uploaded_file.read())
    img = image.load_img(image_stream, target_size=(128, 128))
    img_rgb = img.convert("RGB")

    # Convert PIL Image to data URL
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_data = base64.b64encode(img_buffer.getvalue()).decode()

    # Preprocess the image
    img_array = image.img_to_array(img_rgb)
    img_array_float32 = img_array.astype('float32')
    img_array_float32 /= 255
    img_array = np.expand_dims(img_array_float32, axis=0)

    # Make predictions using the Keras model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    class_dict = {
        0: 'Non Eligible',
        1: 'Eligible'
       }

    result = class_dict[predicted_class[0]]

    st.write(f"Prediction: {result}")

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
