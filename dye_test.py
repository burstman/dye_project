import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


@st.cache_resource
def load_cached_model():
    return load_model("dye_options.keras")


try:
    model = load_cached_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load the LabelEncoder for 'couleur' (input)
le_couleur = LabelEncoder()
le_couleur.classes_ = np.load("color_mapping.npy", allow_pickle=True)
print(le_couleur.classes_)

available_colors = le_couleur.classes_.tolist()

input_colors = st.selectbox("Select a color:", options=available_colors)

# Load the LabelEncoder for class predictions
le_class = LabelEncoder()
le_class.classes_ = np.load("output.npy", allow_pickle=True)
print(le_class.classes_)

# Streamlit UI
st.title("Garment Dyeing Decision Prediction")

st.write("Or input data manually below:")

# Dropdown menu for color selection
input_couleur = st.selectbox("Choose a color", options=available_colors)
input_data = []

if input_couleur:
    # Encode the input 'couleur' using the LabelEncoder
    encoded_couleur = le_couleur.transform([input_couleur])[0]
    input_data.append(encoded_couleur)

# Add additional features if applicable (you can adjust these as per your input data structure)
input_data.append(st.number_input("Delta_a"))
input_data.append(st.number_input("Delte_b"))
input_data.append(st.number_input("Delta_h"))
input_data.append(st.number_input("Delta_L"))
input_data.append(st.number_input("Delta_E"))
# Convert input data to a DataFrame for prediction
if st.button("Predict"):
    input_df = pd.DataFrame(
        [input_data],
        columns=["couleur", "Delta_a", "Delte_b", "Delta_h", "Delta_L", "Delta_E"],
    )

    # Make predictions
    prediction = model.predict(input_df)
    predicted_class = np.argmax(prediction, axis=1)

    # Convert the encoded labels back to the original class labels
    decoded_prediction = le_class.inverse_transform(predicted_class)

    st.write("Prediction:")
    st.write(decoded_prediction)
