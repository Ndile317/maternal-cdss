import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Maternal Health CDSS", layout="wide")

st.markdown("""
<style>
.risk-low { background: #d4edda; color: #155724; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; }
.risk-medium { background: #fff3cd; color: #856404; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; }
.risk-high { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 10px; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

class MaternalCDSS:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.load_model()
    
    def load_model(self):
        try:
            with open('cdss_model.pkl', 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
        except:
            self.create_demo_model()
    
    def create_demo_model(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(100, 12)
        y = np.random.randint(0, 2, 100)
        self.model.fit(X, y)
        self.feature_names = ['age', 'parity', 'wealth', 'education', 'rural', 'anc_visits', 'first_visit_month', 'adequate_visits', 'late_start', 'anc_service_score', 'teen_pregnancy', 'high_parity']
    
    def predict_risk(self, patient_data):
        try:
            df = pd.DataFrame([patient_data])[self.feature_names]
            return self.model.predict_proba(df)[0, 1]
        except:
            return 0.5

def main():
    st.title("Maternal Health CDSS - Zimbabwe")
    st.write("Clinical Decision Support System")
    
    cdss = MaternalCDSS()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        age = st.slider("Age", 15, 45, 25)
        visits = st.slider("ANC Visits", 0, 12, 4)
        wealth = st.selectbox("Wealth", [1,2,3,4,5], format_func=lambda x: f"{x} - {'Poorest' if x==1 else 'Poorer' if x==2 else 'Middle' if x==3 else 'Richer' if x==4 else 'Richest'}")
    
    with col2:
        st.subheader("ANC Services")
        education = st.selectbox("Education", [0,1,2,3], format_func=lambda x: f"{x} - {'No education' if x==0 else 'Primary' if x==1 else 'Secondary' if x==2 else 'Higher'}")
        rural = st.radio("Residence", [0,1], format_func=lambda x: "Urban" if x==0 else "Rural", horizontal=True)
        services = st.slider("Services Received", 0, 5, 3)
    
    if st.button("Assess Pregnancy Risk", type="primary"):
        patient_data = {
            'age': age, 'parity': 2, 'wealth': wealth, 'education': education, 'rural': rural,
            'anc_visits': visits, 'first_visit_month': 3, 'adequate_visits': 1 if visits >= 4 else 0,
            'late_start': 0, 'anc_service_score': services, 'teen_pregnancy': 1 if age < 20 else 0, 
            'high_parity': 0
        }
        
        risk_score = cdss.predict_risk(patient_data)
        
        st.subheader("Assessment Results")
        
        if risk_score < 0.3:
            risk_class, risk_level, advice = "risk-low", "LOW RISK", "Continue routine care"
        elif risk_score < 0.6:
            risk_class, risk_level, advice = "risk-medium", "MEDIUM RISK", "Increase monitoring"
        else:
            risk_class, risk_level, advice = "risk-high", "HIGH RISK", "Refer to specialist"
        
        st.markdown(f'<div class="{risk_class}"><h3>{risk_level}</h3><p>{advice}</p></div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Risk Score", f"{risk_score:.3f}")
        with col4:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=risk_score, domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 1]}, 'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.6], 'color': "yellow"},
                    {'range': [0.6, 1], 'color': "red"}]}))
            fig.update_layout(height=200, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Recommendations")
        if visits < 4: st.write("- Schedule more ANC visits")
        if services < 3: st.write("- Complete essential services")
        if wealth <= 2: st.write("- Connect with support services")

if __name__ == "__main__":
    main()
