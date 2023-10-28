mport streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import base64
import io
from PIL import Image
# Load the Keras model
from tensorflow.keras.models import load_model

st.title("Voter's Eligibility Checker")
image = Image.open("ELECTION AI.jpg")
st.image(image, use_column_width=True)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
img_data = None

if uploaded_file is not None:
    # Use in-memory image
    image_stream = io.BytesIO(uploaded_file.read())
    img = Image.open(image_stream, target_size=(128, 128))
    img_rgb = img.convert("RGB")

    # Convert PIL Image to data URL
    img_buffer = io.BytesIO()
    img_rgb.save(img_buffer, format="PNG")
    img_data = base64.b64encode(img_buffer.getvalue()).decode()

    # Preprocess the image
    img_array = image.img_to_array(img_rgb)
    img_array_float32 = img_array.astype('float32')
    img_array_float32 /= 255
    img_array = np.expand_dims(img_array_float32, axis=0)

    # Load the pre-trained Keras model
    model = load_model('best_model.h5')  # Replace 'best_model.h5' with your actual model filename

    # Make predictions using the Keras model
    predictions = model.predict(img_array)
    predicted_class = 1 if predictions[0][0] >= 0.73 else 0


    if predicted_class == 1:
        result = "Eligible"
    else:
        result = "Non Eligible"

    st.write(f"This Citizen is: {result}")

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
