import streamlit as st
import pandas as pd
import joblib
import requests
from sqlalchemy import create_engine

st.set_page_config(page_title="Enterprise Dashbord", layout="wide")
st.title("📊 Full Stack Enterprise System.")

st.subheader("Live Database Connected ....")
st.subheader("1. Live Database Connection")
try:
    # Your connection engine
    engine = create_engine("sqlite:///local_fallback.db") 
    st.success("✅ Connected to Database Engine")
    
    # ---> NEW CODE START <---
    # We ask Pandas to run the SQL query and fetch the table
    try:
        df = pd.read_sql("SELECT * FROM Inventory", engine)
        
        # We ask Streamlit to draw the table beautifully on the web
        st.write("📦 **Live Pharmacy Inventory:**")
        st.dataframe(df, use_container_width=True)
        
    except Exception as table_error:
        # If the database is completely brand new and has no tables yet
        st.info("Database connected, but the 'Inventory' table is empty or missing. You need to run an INSERT script to add medicines!")
    # ---> NEW CODE END <---

except Exception as e:
    st.error(f"Database connection failed: {e}")


st.divider()


st.subheader("2. Patient Risk Predictor (Machine Learning)")


try:
    model = joblib.load('patient_model.pkl')
    
    
    col1, col2, col3 = st.columns(3)
    age = col1.number_input("Patient Age", 18, 100, 50)
    dist = col2.number_input("Distance to Clinic (km)", 0, 100, 10)
    missed = col3.number_input("Missed Appointments", 0, 10, 0)
    
    if st.button("Predict Risk"):
        
        prediction = model.predict([[age, dist, missed]])
        if prediction[0] == 1:
            st.error("⚠️ HIGH RISK: Patient is likely to churn/drop out.")
        else:
            st.success("✅ LOW RISK: Patient is stable.")
            
except FileNotFoundError:
    st.warning("Please run train_model.py first to generate the .pkl file.")

st.divider()

st.subheader("3. Live External Data (REST API)")
search_term = st.text_input("Search Global Medication Database (e.g., Aspirin)")

if st.button("Search API"):
    url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{search_term}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        try:
            brand_name = data['results'][0]['openfda']['brand_name'][0]
            st.write(f"**Found Brand:** {brand_name}")
            st.info(data['results'][0]['warnings'][0])
        except KeyError:
            st.warning("Data found, but warning labels are missing for this item.")
    else:
        st.error("Item not found in the external database.")