import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 🚀 Streamlit page settings
st.set_page_config(page_title="Food Calorie Estimator", layout="centered", page_icon="🍽️")

# ✅ Load model and calories.csv
@st.cache_resource
def load_model_and_data():
    model = load_model("food_classifier_model.h5")
    calories = pd.read_csv("calories.csv")  # must have 'food' and 'calories' columns
    return model, calories

model, calorie_df = load_model_and_data()

# 🔁 Get class labels (sorted by folder/class index order)
class_labels = list(calorie_df['food'])

# 📸 Predict function
def predict_food(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    predicted_food = class_labels[predicted_index]
    predicted_cal = calorie_df.loc[calorie_df['food'] == predicted_food, 'calories'].values[0]
    return predicted_food, predicted_cal

# 🌟 App title and description
st.markdown("<h1 style='text-align:center; color:#ff6f00;'>🍱 Food Calorie Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Upload a food image to get its name and estimated calories.</p>", unsafe_allow_html=True)

# 📤 File uploader
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

# 📷 Show and predict
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        food_name, calories = predict_food(img)

    st.success(f"🍽️ **Predicted Food:** {food_name.capitalize()}")
    st.info(f"🔥 **Estimated Calories:** {calories} kcal")

    # 🍩 Calorie pie chart
    fig, ax = plt.subplots()
    ax.pie([calories, 2500 - calories], labels=[food_name, "Remaining (from 2500 kcal)"], colors=["#ff9800", "#e0e0e0"], startangle=90, autopct='%1.1f%%')
    ax.axis("equal")
    st.pyplot(fig)

else:
    st.warning("📤 Please upload an image to start prediction.")

# ✨ Footer
st.markdown("<hr><p style='text-align: center; color: gray;'>Made with ❤️ by Sabreena</p>", unsafe_allow_html=True)

