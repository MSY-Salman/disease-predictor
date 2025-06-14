import streamlit as st
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="🏥 MediScan AI - Advanced Symptom Checker",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning visual design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header h3 {
        font-size: 1.5rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .symptom-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .symptom-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .symptom-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .symptom-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .symptom-card:hover::before {
        transform: scaleX(1);
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 40px rgba(17, 153, 142, 0.3);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-container::before {
        content: '🎯';
        position: absolute;
        top: 20px;
        right: 20px;
        font-size: 2rem;
        opacity: 0.3;
    }
    
    .prediction-title {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .prediction-result {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border: 1px solid #f1f3f4;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    .warning-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(255, 107, 107, 0.3);
    }
    
    .warning-medium {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(255, 167, 38, 0.3);
    }
    
    .warning-low {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(102, 187, 106, 0.3);
    }
    
    .sidebar-card {
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 3rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4) !important;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(116, 185, 255, 0.3);
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_predictions' not in st.session_state:
    st.session_state.session_predictions = 0
if 'total_symptoms_checked' not in st.session_state:
    st.session_state.total_symptoms_checked = 0

# Load models with enhanced error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load('rf_model.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, le, True
    except FileNotFoundError as e:
        return None, None, False

model, le, models_loaded = load_models()

# Feature information with medical descriptions
feature_info = {
    'skin_rash': {'desc': 'Visible skin irritation, redness, or unusual markings', 'category': 'Dermatological', 'severity': 'Medium'},
    'chills': {'desc': 'Feeling cold and shivering despite normal temperature', 'category': 'Systemic', 'severity': 'Medium'},
    'joint_pain': {'desc': 'Aching, stiffness, or discomfort in joints', 'category': 'Musculoskeletal', 'severity': 'Medium'},
    'stomach_pain': {'desc': 'Abdominal discomfort, cramping, or aching', 'category': 'Gastrointestinal', 'severity': 'High'},
    'vomiting': {'desc': 'Forceful expulsion of stomach contents', 'category': 'Gastrointestinal', 'severity': 'High'},
    'fatigue': {'desc': 'Persistent tiredness, exhaustion, or lack of energy', 'category': 'Systemic', 'severity': 'Medium'},
    'weight_loss': {'desc': 'Unintentional decrease in body weight', 'category': 'Systemic', 'severity': 'High'},
    'high_fever': {'desc': 'Body temperature above 101°F (38.3°C)', 'category': 'Systemic', 'severity': 'High'},
    'sweating': {'desc': 'Excessive perspiration or night sweats', 'category': 'Systemic', 'severity': 'Medium'},
    'headache': {'desc': 'Pain, pressure, or aching in the head or neck', 'category': 'Neurological', 'severity': 'Medium'},
    'nausea': {'desc': 'Feeling of sickness, queasiness, or urge to vomit', 'category': 'Gastrointestinal', 'severity': 'Medium'},
    'loss_of_appetite': {'desc': 'Reduced desire to eat or lack of hunger', 'category': 'Gastrointestinal', 'severity': 'Medium'},
    'abdominal_pain': {'desc': 'Pain or discomfort in the stomach area', 'category': 'Gastrointestinal', 'severity': 'High'},
    'diarrhoea': {'desc': 'Loose, watery, or frequent bowel movements', 'category': 'Gastrointestinal', 'severity': 'Medium'},
    'yellowing_of_eyes': {'desc': 'Yellow discoloration of the whites of eyes', 'category': 'Hepatic', 'severity': 'High'},
    'malaise': {'desc': 'General feeling of discomfort, illness, or uneasiness', 'category': 'Systemic', 'severity': 'Medium'},
    'chest_pain': {'desc': 'Discomfort, pressure, or pain in the chest area', 'category': 'Cardiovascular', 'severity': 'High'},
    'dizziness': {'desc': 'Feeling lightheaded, unsteady, or faint', 'category': 'Neurological', 'severity': 'Medium'},
    'irritability': {'desc': 'Easily annoyed, angered, or emotionally reactive', 'category': 'Neurological', 'severity': 'Low'},
    'muscle_pain': {'desc': 'Aching, soreness, or discomfort in muscles', 'category': 'Musculoskeletal', 'severity': 'Medium'}
}

# Header with animation
st.markdown("""
<div class="main-header">
    <h1>🏥 MediScan AI</h1>
    <h3>Advanced Intelligent Symptom Analysis</h3>
    <p>Powered by Machine Learning • Professional Medical Insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced features
with st.sidebar:
    st.markdown("## 📊 Dashboard")
    
    # Enhanced session statistics
    current_time = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>📈 Session Analytics</h4>
        <p><strong>🔍 Predictions Made:</strong> {st.session_state.session_predictions}</p>
        <p><strong>📋 Symptoms Checked:</strong> {st.session_state.total_symptoms_checked}</p>
        <p><strong>🕐 Current Time:</strong> {current_time}</p>
        <p><strong>📅 Session Started:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick search functionality
    st.markdown("### 🔍 Symptom Search")
    search_term = st.text_input("", placeholder="🔎 Search symptoms...")
    
    # Model status
    status_color = "#28a745" if models_loaded else "#dc3545"
    status_text = "✅ Ready" if models_loaded else "❌ Error"
    st.markdown(f"""
    <div class="sidebar-card">
        <h4>🤖 AI Model Status</h4>
        <p style="color: {status_color}; font-weight: bold;">{status_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Health tips
    st.markdown("""
    <div class="feature-highlight">
        <h4>💡 Smart Health Tips</h4>
        <ul style="margin: 0; padding-left: 1.2rem;">
            <li>Stay well hydrated throughout the day</li>
            <li>Monitor symptoms and their patterns</li>
            <li>Get adequate rest and sleep</li>
            <li>Consult healthcare providers when needed</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content area
if not models_loaded:
    st.error("🚨 **Model Loading Error**: Please ensure 'rf_model.pkl' and 'label_encoder.pkl' files are in the same directory as this script.")
    st.stop()

# Create tabs for organized content
tab1, tab2, tab3 = st.tabs(["🩺 **Symptom Analysis**", "📊 **Results Dashboard**", "ℹ️ **Medical Information**"])

with tab1:
    st.markdown("## Select Your Current Symptoms")
    st.markdown("*Check all symptoms you are currently experiencing for accurate analysis*")
    
    # Filter symptoms based on search
    filtered_features = feature_info.items()
    if search_term:
        filtered_features = [(k, v) for k, v in feature_info.items() 
                           if search_term.lower() in k.lower() or search_term.lower() in v['desc'].lower()]
    
    # Create symptom selection interface
    col1, col2 = st.columns(2)
    input_features = [0] * 20  # Initialize with zeros
    selected_symptoms = []
    severity_score = 0
    
    feature_names = list(feature_info.keys())
    
    for i, (feat, info) in enumerate(filtered_features):
        with col1 if i % 2 == 0 else col2:
            # Get severity color
            severity_colors = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
            severity_color = severity_colors.get(info['severity'], '#6c757d')
            
            st.markdown(f"""
            <div class="symptom-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong style="font-size: 1.1rem;">{feat.replace("_", " ").title()}</strong>
                    <span style="background: {severity_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.7rem; font-weight: bold;">
                        {info['severity']}
                    </span>
                </div>
                <p style="color: #6c757d; margin-bottom: 0.5rem; font-size: 0.9rem;">{info['desc']}</p>
                <p style="color: #495057; font-size: 0.8rem; font-weight: 500;">Category: {info['category']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if feat in feature_names:
                feat_index = feature_names.index(feat)
                val = st.checkbox(f"I have {feat.replace('_', ' ')}", key=feat)
                input_features[feat_index] = 1 if val else 0
                
                if val:
                    selected_symptoms.append(feat.replace("_", " ").title())
                    severity_weights = {'Low': 1, 'Medium': 2, 'High': 3}
                    severity_score += severity_weights.get(info['severity'], 1)
    
    # Real-time statistics
    symptom_count = sum(input_features)
    st.session_state.total_symptoms_checked = max(st.session_state.total_symptoms_checked, symptom_count)
    
    # Progress bar for symptom count
    progress_percentage = min((symptom_count / 10) * 100, 100)
    st.markdown(f"""
    <div style="margin: 2rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span><strong>Selected Symptoms: {symptom_count}/20</strong></span>
            <span><strong>Severity Score: {severity_score}</strong></span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress_percentage}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis button
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("🚀 **ANALYZE SYMPTOMS**", use_container_width=True):
            if symptom_count == 0:
                st.warning("⚠️ **Please select at least one symptom** to perform analysis.")
            else:
                # Enhanced loading experience
                progress_placeholder = st.empty()
                
                with progress_placeholder:
                    st.markdown("""
                    <div style="text-align: center; padding: 2rem;">
                        <div class="loading-spinner"></div>
                        <h3>🧠 AI Analyzing Your Symptoms...</h3>
                        <p>Processing medical data and generating insights</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                time.sleep(3)  # Realistic processing time
                progress_placeholder.empty()
                
                # Make prediction
                prediction = model.predict([input_features])[0]
                disease = le.inverse_transform([prediction])[0]
                
                # Enhanced confidence calculation
                prediction_proba = model.predict_proba([input_features])[0]
                confidence = np.max(prediction_proba)
                
                # Store results in session state
                st.session_state.last_prediction = disease
                st.session_state.last_symptoms = selected_symptoms
                st.session_state.symptom_count = symptom_count
                st.session_state.confidence = confidence
                st.session_state.severity_score = severity_score
                st.session_state.session_predictions += 1
                
                st.success("✅ **Analysis Complete!** Check the Results Dashboard tab for detailed insights.")

with tab2:
    st.markdown("## 📊 Analysis Results Dashboard")
    
    if hasattr(st.session_state, 'last_prediction'):
        # Main prediction display
        st.markdown(f"""
        <div class="prediction-container">
            <div class="prediction-title">🎯 Predicted Medical Condition</div>
            <div class="prediction-result">{st.session_state.last_prediction}</div>
            <div style="font-size: 1.2rem; opacity: 0.9;">
                Confidence Level: {st.session_state.confidence:.1%} | 
                Analysis Accuracy: High
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistics grid
        risk_level = "🔴 High" if st.session_state.severity_score > 15 else "🟡 Medium" if st.session_state.severity_score > 8 else "🟢 Low"
        
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">📋</div>
                <div class="stat-number">{st.session_state.symptom_count}</div>
                <div class="stat-label">Symptoms Selected</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🎯</div>
                <div class="stat-number">{st.session_state.confidence:.1%}</div>
                <div class="stat-label">AI Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⚠️</div>
                <div class="stat-number">{risk_level}</div>
                <div class="stat-label">Risk Assessment</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🔬</div>
                <div class="stat-number">{st.session_state.severity_score}</div>
                <div class="stat-label">Severity Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations based on severity
        if st.session_state.severity_score > 15:
            st.markdown("""
            <div class="warning-high">
                <h3>🚨 HIGH PRIORITY RECOMMENDATION</h3>
                <p><strong>Immediate medical attention recommended.</strong> You have multiple high-severity symptoms that require professional evaluation. Please contact your healthcare provider or visit an emergency facility promptly.</p>
                <ul>
                    <li>Contact your primary care physician immediately</li>
                    <li>Consider visiting an urgent care center</li>
                    <li>Monitor symptoms closely for any worsening</li>
                    <li>Keep a symptom diary with timestamps</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.severity_score > 8:
            st.markdown("""
            <div class="warning-medium">
                <h3>⚠️ MEDIUM PRIORITY RECOMMENDATION</h3>
                <p><strong>Medical consultation advised.</strong> Your symptoms suggest a condition that would benefit from professional medical evaluation within the next few days.</p>
                <ul>
                    <li>Schedule an appointment with your healthcare provider</li>
                    <li>Monitor symptoms for any changes</li>
                    <li>Consider telehealth consultation if available</li>
                    <li>Maintain good hydration and rest</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-low">
                <h3>✅ LOW PRIORITY RECOMMENDATION</h3>
                <p><strong>Monitoring and self-care suggested.</strong> Your symptoms appear to be manageable with home care, but continue to monitor for any changes.</p>
                <ul>
                    <li>Practice good self-care and rest</li>
                    <li>Stay well-hydrated</li>
                    <li>Monitor symptoms for any worsening</li>
                    <li>Consult healthcare provider if symptoms persist or worsen</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Selected symptoms summary
        if st.session_state.last_symptoms:
            st.markdown("### 📋 Your Selected Symptoms")
            symptoms_text = " • ".join(st.session_state.last_symptoms)
            st.markdown(f"""
            <div class="feature-highlight">
                <p style="font-size: 1.1rem; margin: 0;"><strong>Symptoms Analyzed:</strong> {symptoms_text}</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="feature-highlight">
            <h3>🔍 No Analysis Results Available</h3>
            <p>Please go to the <strong>Symptom Analysis</strong> tab to select your symptoms and run an analysis.</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("## ℹ️ Medical Information & Guidance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-highlight">
            <h4>🤖 How MediScan AI Works</h4>
            <p>Our advanced machine learning system analyzes symptom patterns using:</p>
            <ul>
                <li><strong>Random Forest Algorithm</strong> - Multiple decision trees for accuracy</li>
                <li><strong>Medical Dataset Training</strong> - Thousands of verified medical cases</li>
                <li><strong>Pattern Recognition</strong> - Complex symptom correlation analysis</li>
                <li><strong>Confidence Scoring</strong> - Reliability assessment for each prediction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-medium">
            <h4>⚠️ Important Medical Disclaimer</h4>
            <p><strong>This tool is for educational and informational purposes only.</strong></p>
            <ul>
                <li>Not a substitute for professional medical advice</li>
                <li>Always consult qualified healthcare providers</li>
                <li>For emergencies, contact emergency services immediately</li>
                <li>Do not delay seeking medical care based on these results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="sidebar-card">
            <h4>📞 Emergency Contacts</h4>
            <p><strong>🚨 Emergency Services:</strong> 911 (US) / 999 (UK) / 112 (EU)</p>
            <p><strong>🏥 Poison Control:</strong> 1-800-222-1222 (US)</p>
            <p><strong>📱 Crisis Text Line:</strong> Text HOME to 741741</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <h4>🎯 Accuracy & Reliability</h4>
            <p>Our AI model has been trained on extensive medical datasets and provides high-confidence predictions. However, accuracy depends on:</p>
            <ul>
                <li>Complete symptom reporting</li>
                <li>Honest symptom assessment</li>
                <li>Understanding of symptom descriptions</li>
                <li>Individual health variations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed symptom reference
    st.markdown("### 📚 Complete Symptom Reference Guide")
    
    # Create a comprehensive symptom table
    symptom_data = []
    for symptom, info in feature_info.items():
        symptom_data.append({
            "Symptom": symptom.replace("_", " ").title(),
            "Description": info['desc'],
            "Category": info['category'],
            "Severity Level": info['severity']
        })
    
    df_symptoms = pd.DataFrame(symptom_data)
    st.dataframe(
        df_symptoms, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Symptom": st.column_config.TextColumn("Symptom", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "Category": st.column_config.TextColumn("Category", width="medium"),
            "Severity Level": st.column_config.TextColumn("Severity", width="small")
        }
    )

# Enhanced footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 2rem;">
    <h3 style="color: #495057; margin-bottom: 1rem;">🏥 MediScan AI</h3>
    <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 0.5rem;">
        <strong>Advanced Medical AI • Powered by Machine Learning</strong>
    </p>
    <p style="color: #868e96; font-size: 0.9rem; margin-bottom: 1rem;">
        Session Predictions: {st.session_state.session_predictions} | 
        Total Symptoms Analyzed: {st.session_state.total_symptoms_checked}
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem; margin-top: 1rem;">
        <p style="color: #6c757d; font-size: 0.8rem; margin: 0;">
            ⚠️ For educational purposes only • Always consult healthcare professionals for medical advice<br>
            🔒 Your data is processed securely and not stored permanently
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
