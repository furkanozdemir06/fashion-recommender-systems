import os
import pickle as pkl
import numpy as np
import streamlit as st
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

st.title("Similar Product Finder")

@st.cache_resource
def load_model():
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    return Sequential([base, GlobalMaxPooling2D()])

@st.cache_data
def load_data():
    return np.array(pkl.load(open("Images_features.pkl", "rb"))), pkl.load(open("filenames.pkl", "rb"))

def extract_features(path, model):
    img = image.img_to_array(image.load_img(path, target_size=(224, 224)))
    res = model.predict(preprocess_input(np.expand_dims(img, 0)), verbose=0).flatten()
    return res / norm(res)

model, (features, filenames) = load_model(), load_data()
nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(features)

file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if file:
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())

    st.image(path, caption="Uploaded Image", width=200)
    _, idxs = nn.kneighbors([extract_features(path, model)])

    st.subheader("Similar Products")
    cols = st.columns(5)
    for col, idx in zip(cols, idxs[0]):
        col.image(filenames[idx])