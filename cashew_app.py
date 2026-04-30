from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image


IMG_SIZE = (224, 224)
MODEL_PATH = Path("cashew_model.h5")

# Harus sama dengan urutan CLASS_FOLDERS di file training.
CLASS_NAMES = ["healthy", "anthracnose", "leaf miner", "red rust"]


st.set_page_config(
    page_title="Klasifikasi Daun Jambu Mete",
    layout="centered",
)


@st.cache_resource
def load_cashew_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def prepare_image(uploaded_image: Image.Image) -> np.ndarray:
    img = uploaded_image.convert("RGB").resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def predict(uploaded_image: Image.Image, model):
    prepared_image = prepare_image(uploaded_image)
    probabilities = model.predict(prepared_image, verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    return CLASS_NAMES[best_index], float(probabilities[best_index]), probabilities


def plot_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2e7d32" if score == max(probabilities) else "#78909c" for score in probabilities]
    ax.barh(CLASS_NAMES, probabilities * 100, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probabilitas (%)")
    ax.set_title("Probabilitas Tiap Kelas")
    ax.grid(axis="x", alpha=0.2)
    for index, score in enumerate(probabilities):
        ax.text(score * 100 + 1, index, f"{score * 100:.2f}%", va="center")
    fig.tight_layout()
    return fig


st.title("Klasifikasi Penyakit Daun Jambu Mete")
st.caption("ResNet50 Transfer Learning")

if not MODEL_PATH.exists():
    st.warning(
        "File model `cashew_model.h5` belum ditemukan. "
        "Jalankan training dulu dengan `python cashew_resnet50_interactive.py`."
    )
    st.stop()

model = load_cashew_model(str(MODEL_PATH))

uploaded_file = st.file_uploader(
    "Pilih gambar daun jambu mete",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Pilih gambar daun untuk menampilkan prediksi.")
    st.stop()

uploaded_image = Image.open(uploaded_file)
st.image(uploaded_image, caption="Gambar yang diupload", use_container_width=True)

if st.button("Prediksi", type="primary", use_container_width=True):
    with st.spinner("Menganalisis gambar..."):
        predicted_class, confidence, probabilities = predict(uploaded_image, model)

    st.subheader("Hasil Prediksi")
    st.success(f"{predicted_class.title()} - confidence {confidence * 100:.2f}%")

    st.pyplot(plot_probabilities(probabilities))

    with st.expander("Detail probabilitas"):
        for class_name, score in zip(CLASS_NAMES, probabilities):
            st.write(f"**{class_name.title()}**: {score * 100:.2f}%")
