
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai

st.set_page_config(page_title="Tata Steel Machine Failure Predictor", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

rf_model, scaler = load_assets()

st.sidebar.header("🔑 GenAI Diagnostics Setup")
gemini_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

st.title("⚙️ Tata Steel: Machine Failure Prediction")
st.write("Input real-time sensor parameters to predict overall Machine Failure risk.")

col1, col2 = st.columns(2)

with col1:
    prod_type = st.selectbox("Product Quality Type", ["Low (L)", "Medium (M)", "High (H)"])
    air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0)
    process_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=320.0, value=310.0)

with col2:
    speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500)
    torque = st.number_input("Torque [Nm]", min_value=10.0, max_value=100.0, value=40.0)
    tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=300, value=100)

if st.button("Predict Machine Failure Risk", type="primary"):
    type_code = {'Low (L)': 0, 'Medium (M)': 1, 'High (H)': 2}[prod_type]
    temp_diff = process_temp - air_temp
    power_load = speed * torque
    
    scaled_feats = scaler.transform([[air_temp, process_temp, speed, torque, tool_wear, temp_diff, power_load]])[0]
    
    input_df = pd.DataFrame([{
        'Type': type_code,
        'Air temperature [K]': scaled_feats[0],
        'Process temperature [K]': scaled_feats[1],
        'Rotational speed [rpm]': scaled_feats[2],
        'Torque [Nm]': scaled_feats[3],
        'Tool wear [min]': scaled_feats[4],
        'Temp_Diff': scaled_feats[5],
        'Power_Load': scaled_feats[6]
    }])
    
    failure_prob = rf_model.predict_proba(input_df)[0][1]
    is_failure = 1 if failure_prob >= 0.5 else 0
    
    st.divider()
    
    if is_failure == 1:
        st.error(f"🚨 **PREDICTION: MACHINE FAILURE LIKELY** (Probability: {failure_prob:.2%})")
    else:
        st.success(f"✅ **PREDICTION: NORMAL OPERATION** (Failure Probability: {failure_prob:.2%})")

    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            gen_model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Act as a reliability engineer at Tata Steel.
            A machine running with these parameters:
            - Quality Type: {prod_type}
            - Air Temp: {air_temp} K
            - Process Temp: {process_temp} K (Diff: {temp_diff:.1f} K)
            - Speed: {speed} RPM
            - Torque: {torque} Nm
            - Tool Wear: {tool_wear} min
            
            Model Risk Output: {'FAILURE LIKELY' if is_failure == 1 else 'NORMAL OPERATION'} ({failure_prob:.2%} probability).
            
            Provide a short 3-bullet breakdown explaining:
            1. Which parameters contribute most to this risk.
            2. Probable failure mechanism (e.g., Heat Dissipation, Overstrain, Tool Wear).
            3. Recommended action for the plant supervisor.
            """
            
            with st.spinner("Generating GenAI Diagnostic Report..."):
                response = gen_model.generate_content(prompt)
                st.markdown("### 🤖 Gemini AI Diagnostic Report")
                st.write(response.text)
        except Exception as e:
            st.warning(f"Gemini API Error: {e}")