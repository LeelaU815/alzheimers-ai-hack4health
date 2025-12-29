import streamlit as st # type: ignore

import streamlit as st

import textwrap

st.markdown("""
<style>
.card-link{
  text-decoration:none !important;
  color:inherit !important;
}

.model-card{
  border:1px solid #e6e6e6;
  border-radius:16px;
  padding:18px;
  cursor:pointer;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  height: 100%;
}

.model-card:hover{
  transform: translateY(-3px);
  box-shadow: 0 10px 26px rgba(0,0,0,.08);
  border-color:#cfcfcf;
}

.model-card .btn-row{
  display:flex;
  align-items:center;
  gap:8px;
  margin-top:14px;
  font-weight:600;
}

.model-card .arrow{
  transition: transform .18s ease;
}

.model-card:hover .arrow{
  transform: translateX(6px);
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="Alzheimer's Prediction Dashboard",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

def model_card(title, desc, href):
    st.markdown(
        textwrap.dedent(f"""
        <a class="card-link" href="{href}">
          <div class="model-card">
            <h3 style="margin:0 0 6px 0;">{title}</h3>
            <div style="opacity:.75; font-size:0.95rem;">{desc}</div>

            
          </div>
        </a>
        """),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    st.title("Alzheimer's Prediction Dashboard")
    st.markdown("Welcome to the Alzheimer's Prediction Dashboard!")
    st.markdown("Below there are multiple tools that *you* can use to predict the likelihood of developing Alzheimer's disease based on various factors such as age, gender, and cognitive test scores.")

    c1, c2= st.columns(2)

    with c1:
        model_card("Clinical Model", "This model utilizes clinical data points such as biomarkers, MMSE scores, and demographic information to classify the patient between normal, mild cognitive impairment, and Alzheimer's disease.", "/Clinical")
    with c2:
        model_card("MRI Model", "This model takes an MRI scan image and predicts the likelihood of Alzheimer's disease. MRI scans needed are from the axial view, converted to PNG or JPG format.", "/MRI")