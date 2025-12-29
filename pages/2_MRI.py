from turtle import pd
import streamlit as st

st.set_page_config(
    page_title="MRI Model",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="Preparing the model...")
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model('models/mri_model/alzheimers_mri_model.keras')
    return model

# @st.cache_data(show_spinner="Preprocessing the image...")
def preprocess_image(image_bytes):
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((128, 128))
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# @st.cache_data(show_spinner="Making the prediction...")
def predict(model, img):
    prediction = model.predict(img)
    print(prediction)
    class_names = [
        "Mild Demented",
        "Moderate Demented",
        "Non Demented",
        "Very Mild Demented"
    ]

    pred = prediction[0]
    predicted_class = pred.argmax()
    confidence = pred[predicted_class]*100
    return pred, class_names[predicted_class], confidence

def display_image(uploaded_file):
    from PIL import Image
    img_display = Image.open(uploaded_file)
    return img_display

if __name__ == "__main__":

    st.title("MRI Model")

    st.write("This model is designed to predict the presence of Alzheimer's disease based on MRI scans. The model uses a Convolutional Neural Network (CNN) architecture and has been trained on a dataset of MRI scans from patients with and without Alzheimer's disease.")
    st.write("To use this model, you can upload an MRI scan image and the model will predict the probability of the patient having Alzheimer's disease.")
    st.write("Please note that this model is for educational and research purposes only and should not be used for clinical decision-making.")
    st.write("When uploading an image, please select the axial view of the MRI scan image.")


    with st.expander("Model details"):
        st.markdown("""
            **Architecture:** Convolutional Neural Network (CNN)  
            **Convolution layers:** 3  
            **Input format:** 128 × 128 grayscale axial MRI slice  
            **Output classes:**  
            • Non Demented  
            • Very Mild Demented  
            • Mild Demented  
            • Moderate Demented  

            **Training approach:**  
            • Data augmentation (Gaussian noise)  
            • Label smoothing for probability calibration  

            **Output type:**  
            • Calibrated softmax probabilities  

            **Intended use:**  
            Educational & research decision‑support only  
            Not approved for clinical diagnosis
            """)

    st.subheader("Select Axial MRI Slice")
    results = st.empty()
    with st.form("image_form"):
        uploaded_file = st.file_uploader("Choose an MRI scan image file", type=["jpg", "jpeg", "png"])
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            results = st.empty()
            if uploaded_file is not None:
                with results.container():
                    print("Starting prediction process...")
                    from PIL import Image
                    print("Showing image...")
                    image_bytes = uploaded_file.getvalue()
                    print("Image loaded.")
                    img_display = display_image(uploaded_file)
                    st.image(img_display, caption="Uploaded MRI Scan Image", use_column_width=True)
                    print("Loading model...")
                    model = load_model()
                    print("Preprocessing image...")
                    img = preprocess_image(image_bytes)
                    print("Making prediction...")
                    pred, label, confidence = predict(model, img)

                    st.markdown("## MRI Analysis Report")

                    st.markdown(f"### Primary Classification: **{label}**")
                    st.markdown(f"**Model Confidence:** {confidence:.2f}%")
                    uncertainty = 1-pred.max()
                    if pred.max() < 0.6:
                        st.warning("Low diagnostic confidence — scan may be ambiguous or low quality.")

                    if uncertainty < 0.2:
                        st.markdown("**Uncertainty:** Low")
                    elif 0.2 <= uncertainty < 0.5:
                        st.markdown("**Uncertainty:** Medium")
                    else:
                        st.markdown("**Uncertainty:** High")

                    class_names = [
                        "Mild Demented",
                        "Moderate Demented",
                        "Non Demented",
                        "Very Mild Demented"
                    ]
                    import numpy as np
                    top2 = np.argsort(pred)[-2:]
                    st.write(f"**Secondary possibility:** {class_names[top2[0]]} ({pred[top2[0]]*100:.1f}%)")

                    if confidence > 80:
                        st.success("High confidence classification")
                    elif confidence > 60:
                        st.warning("Moderate confidence classification")
                    else:
                        st.error("Low confidence — ambiguous scan")

                    
                    import pandas as pd
                    df = pd.DataFrame({
                        "Class": class_names,
                        "Likelihood": pred
                    }).sort_values("Likelihood", ascending=False).set_index("Class")

                    st.dataframe(df.style.format({"Likelihood": "{:.3f}"}))
                    st.bar_chart(df)

                    print("Prediction complete.")
            else:
                st.write("No image uploaded.")