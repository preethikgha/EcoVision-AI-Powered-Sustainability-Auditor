import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import cv2


@st.cache_resource
def load_mobilenet_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])

    model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dense(1, activation='sigmoid')
    ])

    model.build((None, 224, 224, 3))
    model.load_weights("mobilenet_final_v1.weights.h5")

    
    if isinstance(model.layers[0], tf.keras.Sequential):
        for lyr in model.layers[0].layers:
            lyr.trainable = False
        model.layers[0](tf.zeros((1, 224, 224, 3)), training=False)

    return model



def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image)

    if img_array.shape[-1] == 4:  
        img_array = img_array[..., :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array



def extract_surface_features(image_pil):
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_small = cv2.resize(img, (224, 224))

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)


    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    reflectance_metric = np.sum(thresh == 255) / (224 * 224)

    
    edges = cv2.Canny(gray, 70, 140)
    edge_density_factor = np.sum(edges > 0) / (224 * 224)

 
    texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()


    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    saturation_metric = hsv[..., 1].mean()

    return reflectance_metric, edge_density_factor, texture_variance, saturation_metric



def compute_material_profile(features):
    reflectance, edge_density, texture_var, saturation = features

    material_risk_index = 0

    if reflectance > 0.06:
        material_risk_index += 1
    if edge_density < 0.03:
        material_risk_index += 1
    if texture_var < 80:
        material_risk_index += 1
    if saturation < 35:
        material_risk_index += 1

    
    if material_risk_index >= 2:
        return "audit_non_eco"   

    if reflectance < 0.015 and texture_var > 120:
        return "audit_eco"       

    return "none"



st.set_page_config(page_title="EcoVision Classifier", layout="centered")
st.title(" EcoVision — AI-Powered Sustainability Auditor")
st.write("Upload an image to audit its eco-friendliness using AI + surface-level analysis.")

model = load_mobilenet_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    model_pred = model.predict(img_array)[0][0]

    features = extract_surface_features(image)
    material_flag = compute_material_profile(features)

   
    if material_flag == "audit_non_eco":
        st.error("Prediction: Not Eco-Friendly ")
    elif material_flag == "audit_eco":
        st.success("Prediction: Eco-Friendly ")
    else:
        if model_pred >= 0.5:
            st.success(f"Prediction: Eco-Friendly  ({model_pred:.2f})")
        else:
            st.error(f"Prediction: Not Eco-Friendly ({model_pred:.2f})")
