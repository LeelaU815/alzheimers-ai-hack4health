import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Clinical Model",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

feature_names = ['AMY_BIOMARKER','TAU_BIOMARKER','MMSE_TOTAL','CDR_SB','SEX','EDUC']

label_names = {
    0: "Cognitively Normal",
    1: "Mild Cognitive Impairment",
    2: "Alzheimer's Disease"
}

@st.cache_resource(show_spinner="Preparing the model...")
def load_model():
    
    model = xgb.Booster()
    model.load_model("models/clinical_model/alzheimers_clinical_model.json")
    return model

def predict(_model, input_df):
    dnew = xgb.DMatrix(input_df)
    probs = _model.predict(dnew, iteration_range=(0, _model.best_iteration+1))[0]
    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return pred_class, probs, confidence

@st.cache_data
def extract(x):
    return x.split(' - ')[0]

if __name__ == "__main__":
    st.title("Clinical Model")

    st.write("This is a simple clinical model that uses machine learning algorithms to predict patient outcomes based on various clinical data points. The model can be trained on a dataset of patient records and used to make predictions on new, unseen data.")
    st.write("To use the model, please fill in the required fields below and click the 'Predict' button. The model will then output the predicted outcome for the given patient.")
    st.write("Please note that the accuracy of the model depends on the quality and quantity of the training data, as well as the specific clinical data points used as input.")

    with st.expander("Model details"):
        st.markdown("""
        **Model:** XGBoost multiclass classifier  
        **Inputs:** AMY_BIOMARKER, TAU_BIOMARKER, MMSE_TOTAL, CDR_SB, SEX, EDUC  
        **Outputs:** Cognitively Normal / Mild Cognitive Impairment / Alzheimer’s Disease  
        **Notes:** Output values are model likelihoods (not a clinical diagnosis).  
        **Use:** Educational & research decision-support only.
        """)
    
    with st.form("clinical_model_form"):
        st.subheader("Input Clinical Data")
        amy = st.number_input("Amyloid Biomarker (AMY_BIOMARKER)",value=0.0, help="Unit depends on dataset assay.")
        tau = st.number_input("Tau Biomarker (TAU_BIOMARKER)",value=0.0,help="Unit depends on dataset assay.")
        mmse = st.number_input("Mini-Mental State Examination Total Score (MMSE_TOTAL)",value=0,help="Range: 0-30")
        cdr = st.number_input("Clinical Dementia Rating Scale (CDR_SB)",value=0.0,help="Typical range: 0.0-18.0")
        sex = st.selectbox("Sex", ["Male", "Female"],help="Biological")
        educ = st.number_input("Years of Education (EDUC)",value=0,help='Years of education (common range 0-30)')
        submitted = st.form_submit_button("Predict")

        if submitted:
            sex = extract(sex)
            if sex=="Male":
                sex = 0
            else:
                sex = 1
            x_pred = [
                amy,
                tau,
                mmse,
                cdr,
                sex,
                educ
            ]
            x_pred = pd.DataFrame([x_pred], columns=feature_names)
            model = load_model()
            pred_class, probs, confidence = predict(model, x_pred)

            probs = np.array(probs, dtype=float)
            order = np.argsort(probs)[::-1]
            primary = int(order[0])
            secondary = int(order[1])
            uncertainty = 1.0 - float(probs[primary])

            st.markdown("## AI-Generated Clinical Risk Summary")
            st.markdown(f"### Primary outcome: **{label_names[primary]}**")
            st.caption(
                f"Model confidence: **{probs[primary]*100:.2f}%**  •  "
                f"Uncertainty: **{'Low' if uncertainty < 0.20 else 'Medium' if uncertainty < 0.40 else 'High'}**"
            )
            st.caption(f"Secondary possibility: **{label_names[secondary]}** ({probs[secondary]*100:.1f}%)")

            if probs[primary] >= 0.80:
                st.success("High confidence classification")
            elif probs[primary] >= 0.60:
                st.warning("Moderate confidence classification")
            else:
                st.error("Low confidence — ambiguous inputs for this model")

            
            df = pd.DataFrame({
                "Diagnosis": [label_names[i] for i in range(len(probs))],
                "Model Likelihood": probs
            }).sort_values("Model Likelihood", ascending=False).set_index("Diagnosis")

            st.dataframe(df.style.format({"Model Likelihood": "{:.3f}"}), use_container_width=True)
            st.bar_chart(df)


