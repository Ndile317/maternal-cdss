import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Maternal Health CDSS - Zimbabwe",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #4A5568;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .section-header {
        font-size: 1.6rem;
        color: #2D3748;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 8px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-low { 
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(76, 175, 80, 0.3);
    }
    .risk-medium { 
        background: linear-gradient(135deg, #FF9800 0%, #FFC107 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(255, 152, 0, 0.3);
    }
    .risk-high { 
        background: linear-gradient(135deg, #F44336 0%, #E91E63 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(244, 67, 54, 0.3);
    }
    .info-box {
        background: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .recommendation-card {
        background: white;
        border: 2px solid #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

class ResearchCDSS:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.feature_descriptions = {}
        self.load_model()
    
    def load_model(self):
        try:
            with open('research_model.pkl', 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.feature_descriptions = data.get('feature_descriptions', {})
            st.sidebar.success("Research-Based CDSS Model Successfully Loaded")
        except Exception as e:
            st.sidebar.error(f"Model loading error: {e}")
            self.create_fallback_model()
    
    def create_fallback_model(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        self.model.fit(X, y)
        self.feature_names = [
            'anc_quality_good', 'wealth_anc_interaction', 'anc_service_score', 
            'm42d_1', 'anc_first_trimester', 'm45_1', 'v025', 
            'anc_adequate_visits', 'm14_1', 'v003'
        ]
        st.sidebar.warning("Using Demonstration Model for Testing")
    
    def predict_risk(self, patient_data):
        try:
            df = pd.DataFrame([patient_data])[self.feature_names]
            risk_score = self.model.predict_proba(df)[0, 1]
            return risk_score
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return 0.5

def main():
    # Header Section
    st.markdown('<h1 class="main-header">Maternal Health CDSS</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Zimbabwe Clinical Decision Support System</div>', unsafe_allow_html=True)
    
    # Hero Section
    col_hero1, col_hero2, col_hero3 = st.columns(3)
    with col_hero1:
        st.metric("Research Based", "10 SHAP Features")
    with col_hero2:
        st.metric("Model Performance", "AUC 0.7735")
    with col_hero3:
        st.metric("Clinical Utility", "High")
    
    st.markdown("---")
    
    cdss = ResearchCDSS()
    
    # Research Information Sidebar
    with st.sidebar:
        st.markdown("## Research Dashboard")
        st.markdown("---")
        
        st.markdown("### Model Information")
        st.markdown("""
        - **Algorithm**: Random Forest
        - **Features**: 10 Top Predictors
        - **Data Source**: Zimbabwe DHS 2015
        - **Validation**: Cross-Validated
        """)
        
        st.markdown("### Top 5 SHAP Features")
        features = [
            "ANC Quality Good (13.4%)",
            "Wealth-ANC Interaction (12.2%)", 
            "ANC Service Score (10.8%)",
            "Urine Testing (10.4%)",
            "First Trimester ANC (8.9%)"
        ]
        for feature in features:
            st.markdown(f"• {feature}")
        
        st.markdown("---")
        st.markdown("### Clinical Guidelines")
        st.markdown("""
        - Low Risk: < 0.3
        - Medium Risk: 0.3 - 0.6  
        - High Risk: > 0.6
        """)
    
    # Main Assessment Interface
    st.markdown('<div class="section-header">Patient Risk Assessment</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>RESEARCH-BASED ASSESSMENT:</strong> This clinical decision support system utilizes the top 10 predictors 
    identified through SHAP analysis of Zimbabwe maternal health data to provide evidence-based risk stratification.
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form in Tabs for Better Organization
    tab1, tab2, tab3 = st.tabs(["Demographic Information", "ANC Services", "Clinical Factors"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Socioeconomic Factors")
            wealth = st.selectbox(
                "Wealth Quintile", 
                [1, 2, 3, 4, 5],
                format_func=lambda x: f"{x} - {'Poorest' if x==1 else 'Poorer' if x==2 else 'Middle' if x==3 else 'Richer' if x==4 else 'Richest'}"
            )
            
            residence = st.radio(
                "Residence Type",
                [1, 2],
                format_func=lambda x: "Urban" if x == 1 else "Rural",
                horizontal=True
            )
        
        with col2:
            st.markdown("#### Geographic Information")
            region = st.selectbox(
                "Administrative Region",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                help="Select the patient's region of residence"
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ANC Visit Information")
            anc_visits = st.slider("Number of ANC Visits", 0, 12, 4, 
                                 help="Total number of antenatal care visits")
            first_visit_month = st.slider("Month of First ANC Visit", 1, 9, 3,
                                        help="Month of pregnancy when first ANC visit occurred")
        
        with col2:
            st.markdown("#### Essential ANC Services")
            urine_test = st.checkbox("Urine Testing", True, 
                                   help="Urine sample tested for protein and sugar")
            iron_supplements = st.checkbox("Iron Supplementation", True,
                                         help="Received iron folate supplements")
            
            st.markdown("#### Additional Services")
            col2a, col2b = st.columns(2)
            with col2a:
                bp_check = st.checkbox("Blood Pressure", True)
                weight_check = st.checkbox("Weight Measurement", True)
            with col2b:
                blood_test = st.checkbox("Blood Testing", True)
                tetanus = st.checkbox("Tetanus Vaccine", True)
    
    with tab3:
        st.markdown("#### Service Quality Indicators")
        col1, col2, col3 = st.columns(3)
        
        # Calculate derived features
        service_score = sum([bp_check, urine_test, blood_test, iron_supplements, tetanus])
        anc_quality_good = 1 if service_score >= 3 else 0
        wealth_anc_interaction = wealth * service_score
        anc_first_trimester = 1 if first_visit_month <= 3 else 0
        anc_adequate_visits = 1 if anc_visits >= 4 else 0
        
        with col1:
            st.metric("ANC Quality Score", f"{service_score}/5")
            st.metric("Quality Status", "Good" if anc_quality_good else "Needs Improvement")
        
        with col2:
            st.metric("Wealth-Service Interaction", wealth_anc_interaction)
            st.metric("First Trimester ANC", "Yes" if anc_first_trimester else "No")
        
        with col3:
            st.metric("Visit Adequacy", "Adequate" if anc_adequate_visits else "Inadequate")
            st.metric("Residence Type", "Urban" if residence == 1 else "Rural")
    
    # Prepare patient data
    patient_data = {
        'anc_quality_good': anc_quality_good,
        'wealth_anc_interaction': wealth_anc_interaction,
        'anc_service_score': service_score,
        'm42d_1': 1 if urine_test else 0,
        'anc_first_trimester': anc_first_trimester,
        'm45_1': 1 if iron_supplements else 0,
        'v025': residence,
        'anc_adequate_visits': anc_adequate_visits,
        'm14_1': anc_visits,
        'v003': region
    }
    
    # Assessment Button
    st.markdown("---")
    if st.button("CONDUCT COMPREHENSIVE RISK ASSESSMENT", use_container_width=True, type="primary"):
        with st.spinner("Analyzing patient data using research-based predictors..."):
            risk_score = cdss.predict_risk(patient_data)
        
        st.markdown("---")
        st.markdown('<div class="section-header">Clinical Assessment Results</div>', unsafe_allow_html=True)
        
        # Risk Categorization
        if risk_score < 0.3:
            risk_class, risk_level, urgency, icon = "risk-low", "LOW RISK", "Continue Routine Care", "✅"
            color, advice_color = "green", "#4CAF50"
        elif risk_score < 0.6:
            risk_class, risk_level, urgency, icon = "risk-medium", "MEDIUM RISK", "Enhanced Monitoring Recommended", "⚠️"
            color, advice_color = "orange", "#FF9800"
        else:
            risk_class, risk_level, urgency, icon = "risk-high", "HIGH RISK", "Immediate Intervention Required", "🚨"
            color, advice_color = "red", "#F44336"
        
        # Results Display
        col_results1, col_results2 = st.columns([1, 2])
        
        with col_results1:
            # Risk Summary Card
            st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
            st.markdown(f"### {risk_level}")
            st.markdown(f"**Risk Score:** {risk_score:.3f}")
            st.markdown(f"**Clinical Priority:** {urgency}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Key Risk Factors
            st.markdown("#### Identified Risk Factors")
            risk_factors = []
            if not anc_quality_good: 
                risk_factors.append("Inadequate ANC service quality")
            if wealth_anc_interaction < 6: 
                risk_factors.append("Low socioeconomic access to care")
            if service_score < 3: 
                risk_factors.append("Missing essential ANC services")
            if not urine_test: 
                risk_factors.append("No urine testing performed")
            if not anc_first_trimester: 
                risk_factors.append("Late ANC initiation")
            if not anc_adequate_visits: 
                risk_factors.append("Insufficient ANC visits")
            
            for factor in risk_factors:
                st.markdown(f'<div class="recommendation-card">❌ {factor}</div>', unsafe_allow_html=True)
            
            if not risk_factors:
                st.markdown('<div class="recommendation-card">✅ No significant risk factors identified</div>', unsafe_allow_html=True)
        
        with col_results2:
            # Interactive Risk Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={
                    'text': "Maternal Risk Assessment",
                    'font': {'size': 20, 'color': '#2D3748', 'family': "Arial"}
                },
                delta={'reference': 0.5, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [0, 1], 'tickwidth': 2, 'tickcolor': "#2D3748"},
                    'bar': {'color': color, 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.3], 'color': '#C8E6C9'},
                        {'range': [0.3, 0.6], 'color': '#FFECB3'},
                        {'range': [0.6, 1], 'color': '#FFCDD2'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(t=80, b=20, l=20, r=20),
                font={'color': "#2D3748", 'family': "Arial"},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Evidence-Based Recommendations
        st.markdown("---")
        st.markdown('<div class="section-header">Clinical Recommendations</div>', unsafe_allow_html=True)
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("#### Immediate Clinical Actions")
            if risk_score > 0.6:
                st.markdown("""
                <div class="recommendation-card">
                <strong>HIGH PRIORITY ACTIONS:</strong>
                <br>• Refer to specialist maternity care immediately
                <br>• Schedule weekly monitoring appointments
                <br>• Develop comprehensive emergency birth plan
                <br>• Coordinate with multidisciplinary team
                </div>
                """, unsafe_allow_html=True)
            elif risk_score > 0.3:
                st.markdown("""
                <div class="recommendation-card">
                <strong>ENHANCED MONITORING:</strong>
                <br>• Increase ANC visit frequency to bi-weekly
                <br>• Provide targeted health education sessions
                <br>• Assign community health worker for follow-up
                <br>• Monitor for danger signs closely
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommendation-card">
                <strong>ROUTINE CARE:</strong>
                <br>• Continue standard ANC schedule
                <br>• Provide basic health education
                <br>• Standard nutritional counseling
                <br>• Routine danger signs education
                </div>
                """, unsafe_allow_html=True)
        
        with rec_col2:
            st.markdown("#### Modifiable Risk Factors")
            recommendations = []
            if not anc_quality_good:
                recommendations.append("Improve ANC service completeness and quality")
            if service_score < 3:
                recommendations.append("Ensure receipt of all essential ANC services")
            if not anc_first_trimester:
                recommendations.append("Promote early ANC registration and attendance")
            if not anc_adequate_visits:
                recommendations.append("Schedule additional ANC visits to meet guidelines")
            if wealth_anc_interaction < 6:
                recommendations.append("Connect with socioeconomic support services")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f'<div class="recommendation-card">📋 {rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommendation-card">
                <strong>OPTIMAL CARE:</strong> All modifiable factors are within recommended ranges. 
                Continue current care plan with routine monitoring.
                </div>
                """, unsafe_allow_html=True)
        
        # Research Context Footer
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        <strong>RESEARCH CONTEXT:</strong> This clinical decision support system is based on comprehensive 
        analysis of Zimbabwe Demographic Health Survey data using machine learning and SHAP analysis 
        to identify the most significant predictors of adverse maternal outcomes. The model achieves 
        AUC 0.7735 and utilizes the top 10 features for evidence-based risk stratification.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
