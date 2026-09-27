import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Crop Failure Early Warning", page_icon="🌾", layout="wide"
)

# Load model and metadata
model = joblib.load("crop_failure_xgboost_model.pkl")
meta = joblib.load("model_metadata.pkl")

st.title("🌾 Pan-African Crop Failure Early Warning Platform")
st.write(
    "Pre-season AI early-warning system for smallholder maize production."
)
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
  country = st.selectbox("Country", meta["countries"])
  season = st.selectbox("Season", meta["seasons"])
  hybrid = st.selectbox("Hybrid Seed Variety", meta["hybrids"])
with col2:
  prec1 = st.number_input(
      "Early Season Rainfall (mm)", min_value=0.0, value=150.0
  )
  prec2 = st.number_input(
      "Mid Season Rainfall (mm)", min_value=0.0, value=200.0
  )
  prec3 = st.number_input(
      "Late Season Rainfall (mm)", min_value=0.0, value=100.0
  )
with col3:
  n_kg = st.number_input("Nitrogen (N kg/ha)", min_value=0.0, value=30.0)
  p_kg = st.number_input("Phosphorus (P kg/ha)", min_value=0.0, value=15.0)
  k_kg = st.number_input("Potassium (K kg/ha)", min_value=0.0, value=0.0)

st.divider()

if st.button("🔍 ANALYZE FARM RISK", use_container_width=True):
  input_data = pd.DataFrame([{
      "avg_season_gdd": 2500.0,
      "avg_season_ai": 0.8,
      "avg_season_tavg": 22.0,
      "season_prec_1": prec1,
      "season_prec_2": prec2,
      "season_prec_3": prec3,
      "elev": 1200.0,
      "twi": 10.0,
      "soil_rzpawhc": 120.0,
      "soil_clay": 30.0,
      "soil_pH": 6.0,
      "soil_orgC": 1.2,
      "soil_ECEC": 10.0,
      "plant_doy": 100.0,
      "N_kg_ha": n_kg,
      "P_kg_ha": p_kg,
      "K_kg_ha": k_kg,
      "lime_kg_ha": 0.0,
      "country": str(country),
      "season": str(season),
      "hybrid": str(hybrid),
  }])

  probability = model.predict_proba(input_data)[0, 1]
  threshold = 0.20
  risk = "HIGH RISK" if probability >= threshold else "LOW RISK"

  res_col1, res_col2 = st.columns(2)
  res_col1.metric("Predicted Failure Probability", f"{probability:.1%}")
  res_col2.metric("Risk Classification", risk)

  if risk == "HIGH RISK":
    st.error(
        f"⚠️ HIGH RISK: Failure probability exceeds threshold ({threshold:.0%})."
        " Recommend drought mitigation strategies."
    )
  else:
    st.success(
        "✅ LOW RISK: Environmental and input parameters indicate stable yield"
        " conditions."
    )