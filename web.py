# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import pandas as pd
from fpdf import FPDF
import datetime

@tf.keras.utils.register_keras_serializable()
def l2_normalization(x):
    return tf.math.l2_normalize(x, axis=-1)

@tf.keras.utils.register_keras_serializable()
def l2_norm(x):  # Alias in case model refers to this
    return tf.math.l2_normalize(x, axis=-1)

custom_objects = {
    "l2_normalization": l2_normalization,
    "l2_norm": l2_norm
}

tf.keras.config.enable_unsafe_deserialization()
custom_objects = {"l2_normalization": l2_normalization}

@st.cache_resource
def load_models():
    b = load_model("model/binary_model.h5", custom_objects=custom_objects, compile=False)
    be = load_model("model/benign_model.h5", custom_objects=custom_objects, compile=False)
    m = load_model("model/malignant_model.h5", custom_objects=custom_objects, compile=False)
    g = load_model("model/grade_model.h5", custom_objects=custom_objects, compile=False)
    return b, be, m, g

binary_model, benign_model, malignant_model, grade_model = load_models()

binary_labels = ["Benign", "Malignant"]
benign_labels = ["Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma"]
malignant_labels = ["Ductal Carcinoma", "Lobular Carcinoma", "Mucinous Carcinoma", "Papillary Carcinoma"]
grade_labels = ["Grade 1", "Grade 2", "Grade 3"]

def predict_image(p, m, s=(128, 128)):
    i = image.load_img(p, target_size=s)
    a = image.img_to_array(i) / 255.0
    a = np.expand_dims(a, axis=0)
    y = m.predict(a)
    c = np.argmax(y, axis=1)[0]
    return c, y

def generate_pdf(n, a, s, t, r, sub=None):
    p = FPDF()
    p.add_page()
    p.set_font("Arial", size=14)
    p.cell(200, 10, txt="Breast Cancer Diagnosis Report", ln=True, align='C')
    p.set_font("Arial", size=12)
    p.ln(10)
    p.cell(200, 10, txt=f"Patient Name: {n}", ln=True)
    p.cell(200, 10, txt=f"Age: {a}", ln=True)
    p.cell(200, 10, txt=f"Sex: {s}", ln=True)
    p.cell(200, 10, txt=f"Classification Type: {t}", ln=True)
    p.cell(200, 10, txt=f"Prediction: {r}", ln=True)
    if sub:
        p.cell(200, 10, txt=f"Subtype: {sub}", ln=True)
    p.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    os.makedirs("reports", exist_ok=True)
    rp = f"reports/{n}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    p.output(rp)
    return rp

def store_case_data(n, a, s, t, r, sub):
    os.makedirs("records", exist_ok=True)
    d = {
        "Name": n,
        "Age": a,
        "Sex": s,
        "Classification Type": t,
        "Result": r,
        "Subtype": sub,
        "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    df = pd.DataFrame([d])
    x = "records/case_data.xlsx"
    if os.path.exists(x):
        e = pd.read_excel(x)
        u = pd.concat([e, df], ignore_index=True)
        u.to_excel(x, index=False)
    else:
        df.to_excel(x, index=False)

def main():
    st.title("🧬 Breast Cancer Detection")
    st.write("Upload a histopathology image and receive a diagnostic report.")

    n = st.text_input("Patient Name")
    a = st.text_input("Age")
    s = st.selectbox("Sex", ["Male", "Female", "Other"])
    t = st.radio("Choose Classification Type", ["Type Classification", "Grade Classification"])

    up = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if up and n and a and s:
        ip = os.path.join("uploads", up.name)
        os.makedirs("uploads", exist_ok=True)
        with open(ip, "wb") as f:
            f.write(up.getbuffer())

        st.image(up, caption="Uploaded Image",  use_container_width=True)
        st.write("🔍 Processing...")

        r = ""
        sub = ""

        if t == "Type Classification":
            i, _ = predict_image(ip, binary_model)
            r = binary_labels[i]
            st.success(f"Diagnosis: {r}")
            if r == "Benign":
                si, _ = predict_image(ip, benign_model)
                sub = benign_labels[si]
            else:
                si, _ = predict_image(ip, malignant_model)
                sub = malignant_labels[si]
            st.info(f"Subtype: {sub}")

        elif t == "Grade Classification":
            i, _ = predict_image(ip, grade_model)
            r = grade_labels[i]
            st.success(f"Grade: {r}")

        store_case_data(n, a, s, t, r, sub)

        rp = generate_pdf(n, a, s, t, r, sub)
        with open(rp, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name=os.path.basename(rp), mime="application/pdf")

if __name__ == "__main__":
    main()
