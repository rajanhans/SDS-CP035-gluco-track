import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.preprocessing import StandardScaler
import pickle
import io
import base64
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the neural network architecture (MUST match your training model exactly)
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.3):
        super(FFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class DiabetesRiskPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None  # Changed from scaler to preprocessor
        self.feature_names = None
        self.explainer = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            # Load model configuration
            with open('streamlit_app/model_config.pkl', 'rb') as f:
                model_config = pickle.load(f)
            
            # Load feature names
            with open('streamlit_app/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Initialize and load model
            self.model = FFNN(
                input_size=model_config['input_size'],
                hidden_size=model_config.get('hidden_size', 64),
                dropout_rate=model_config.get('dropout_rate', 0.3)
            )
            self.model.load_state_dict(torch.load('streamlit_app/best_model.pth', map_location='cpu'))
            self.model.eval()
            
            # Load preprocessor (not just scaler)
            with open('streamlit_app/preprocessor.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            self.model_loaded = True
            return True
            
        except FileNotFoundError as e:
            st.error(f"Model files not found: {str(e)}")
            st.error("Please upload the following files: best_model.pth, preprocessor.pkl, model_config.pkl, feature_names.pkl")
            return False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    def predict(self, input_data):
        """Make prediction on input data"""
        if not self.model_loaded:
            return None
        
        try:
            # Convert to DataFrame with correct feature order
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data
            
            # DEBUG: Check if all expected features are present
            missing_features = set(self.feature_names) - set(input_df.columns)
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                return None
            
            # Ensure correct column order
            input_df = input_df[self.feature_names]
            
            # DEBUG: Show feature order (remove in production)
            #st.write("Debug - Feature order:", self.feature_names)
            #st.write("Debug - Input order:", input_df.columns.tolist())
            
            # Preprocess the input (this handles scaling of continuous features)
            input_processed = self.preprocessor.transform(input_df)
            
            # Convert to tensor and predict
            input_tensor = torch.FloatTensor(input_processed)
            with torch.no_grad():
                prediction = self.model(input_tensor)
                risk_score = prediction.item()
            
            return risk_score
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
        
# Initialize the predictor
@st.cache_resource
def load_predictor():
    predictor = DiabetesRiskPredictor()
    predictor.load_model()  # This will show errors if files are missing
    return predictor

def create_input_form():
    """Create the user input form"""
    st.sidebar.header("📋 Health Assessment Form")
    st.sidebar.write("Please fill out the following health indicators:")
    
    # Organize inputs into categories
    inputs = {}
    
    # Cardiovascular Health
    st.sidebar.subheader("🫀 Cardiovascular Health")
    inputs['HighBP'] = st.sidebar.selectbox("High Blood Pressure", [0, 1], 
                                          format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['HighChol'] = st.sidebar.selectbox("High Cholesterol", [0, 1],
                                            format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['CholCheck'] = st.sidebar.selectbox("Cholesterol Check (last 5 years)", [0, 1],
                                             format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['Stroke'] = st.sidebar.selectbox("History of Stroke", [0, 1],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['HeartDiseaseorAttack'] = st.sidebar.selectbox("Heart Disease/Attack", [0, 1],
                                                        format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Physical Health
    st.sidebar.subheader("🏃 Physical Health")
    inputs['bmi_category'] = st.sidebar.selectbox("BMI Category", [0, 1, 2, 3, 4, 5],
                                                format_func=lambda x: ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III'][x])
    inputs['PhysActivity'] = st.sidebar.selectbox("Physical Activity (last 30 days)", [0, 1],
                                                format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['DiffWalk'] = st.sidebar.selectbox("Difficulty Walking", [0, 1],
                                            format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['PhysHlth'] = st.sidebar.slider("Physical Health Issues (days in last 30)", 0, 30, 0)
    
    # Lifestyle Factors
    st.sidebar.subheader("🍎 Lifestyle Factors")
    inputs['Smoker'] = st.sidebar.selectbox("Smoker", [0, 1],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['HvyAlcoholConsump'] = st.sidebar.selectbox("Heavy Alcohol Consumption", [0, 1],
                                                     format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['Fruits'] = st.sidebar.selectbox("Daily Fruit Consumption", [0, 1],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['Veggies'] = st.sidebar.selectbox("Daily Vegetable Consumption", [0, 1],
                                           format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Healthcare Access
    st.sidebar.subheader("🏥 Healthcare Access")
    inputs['AnyHealthcare'] = st.sidebar.selectbox("Health Insurance Coverage", [0, 1],
                                                 format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['NoDocbcCost'] = st.sidebar.selectbox("Avoided Doctor Due to Cost", [0, 1],
                                                format_func=lambda x: "No" if x == 0 else "Yes")
    
    # General Health
    st.sidebar.subheader("🌟 General Health")
    inputs['GenHlth'] = st.sidebar.selectbox("General Health", [0, 1, 2, 3, 4],  # Changed to match your encoding
                                           format_func=lambda x: ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'][x])
    inputs['MentHlth'] = st.sidebar.slider("Mental Health Issues (days in last 30)", 0, 30, 0)
    
    # Demographics
    st.sidebar.subheader("👤 Demographics")
    inputs['Sex'] = st.sidebar.selectbox("Sex", [0, 1],
                                       format_func=lambda x: "Female" if x == 0 else "Male")
    # Age group mapping
    age_groups = [
        ("18 to 24", 0),
        ("25 to 29", 1),
        ("30 to 34", 2),
        ("35 to 39", 3),
        ("40 to 44", 4),
        ("45 to 49", 5),
        ("50 to 54", 6),
        ("55 to 59", 7),
        ("60 to 64", 8),
        ("65 to 69", 9),
        ("70 to 74", 10),
        ("75 to 79", 11),
        ("80 or older", 12),
        ("Don't know / Refused / Missing", 13)
    ]
    age_labels = [g[0] for g in age_groups]
    age_codes = [g[1] for g in age_groups]
    selected_age = st.sidebar.selectbox("Age Group", age_codes, format_func=lambda x: age_labels[x])
    inputs['Age'] = selected_age
    # Education mapping
    education_options = [
        ("Never attended school or only kindergarten", 1),
        ("Grades 1 through 8 (Elementary)", 2),
        ("Grades 9 through 11 (Some high school)", 3),
        ("Grade 12 or GED (High school graduate)", 4),
        ("College 1 year to 3 years (Some college or technical school)", 5),
        ("College 4 years or more (College graduate)", 6)
    ]
    education_labels = [e[0] for e in education_options]
    education_codes = [e[1] for e in education_options]
    selected_education = st.sidebar.selectbox("Education Level", education_codes, format_func=lambda x: education_labels[x-1])
    inputs['Education'] = selected_education
    # Income mapping
    income_options = [
        ("Less than $10,000", 1),
        ("$10,000 to less than $15,000", 2),
        ("$15,000 to less than $20,000", 3),
        ("$20,000 to less than $25,000", 4),
        ("$25,000 to less than $35,000", 5),
        ("$35,000 to less than $50,000", 6),
        ("$50,000 to less than $75,000", 7),
        ("More than $75,000", 8)
    ]
    income_labels = [i[0] for i in income_options]
    income_codes = [i[1] for i in income_options]
    selected_income = st.sidebar.selectbox("Income Level", income_codes, format_func=lambda x: income_labels[x-1])
    inputs['Income'] = selected_income
    
    return inputs

def create_risk_visualization(risk_score):
    """Create risk score visualization"""
    # Risk gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Score (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def interpret_risk_score(risk_score):
    """Provide interpretation of the risk score"""
    risk_percentage = risk_score * 100
    
    if risk_percentage < 25:
        risk_level = "Low Risk"
        color = "green"
        recommendations = [
            "Continue maintaining your healthy lifestyle",
            "Regular health check-ups are recommended",
            "Keep up with physical activity and healthy diet"
        ]
    elif risk_percentage < 50:
        risk_level = "Moderate Risk"
        color = "orange"
        recommendations = [
            "Consider lifestyle modifications",
            "Increase physical activity if possible",
            "Monitor diet and reduce processed foods",
            "Regular medical check-ups recommended"
        ]
    elif risk_percentage < 75:
        risk_level = "High Risk"
        color = "red"
        recommendations = [
            "Consult with healthcare provider soon",
            "Consider diabetes screening tests",
            "Implement significant lifestyle changes",
            "Focus on weight management if applicable"
        ]
    else:
        risk_level = "Very High Risk"
        color = "darkred"
        recommendations = [
            "Seek immediate medical consultation",
            "Diabetes screening tests highly recommended",
            "Urgent lifestyle interventions needed",
            "Consider working with diabetes specialist"
        ]
    
    return risk_level, color, recommendations

def main():
    st.title("🩺 Diabetes Risk Assessment Tool")
    st.markdown("### AI-Powered Risk Prediction Based on CDC Health Indicators")
    
    st.markdown("""
    This tool uses a deep learning model trained on CDC BRFSS data to assess diabetes risk based on 
    health, lifestyle, and demographic factors. Please note that this is for informational purposes 
    only and should not replace professional medical advice.
    """)
    
    # Initialize predictor
    predictor = load_predictor()
    
    # Create input form
    user_inputs = create_input_form()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 Assess Diabetes Risk", type="primary", use_container_width=True):
            if predictor.model_loaded:
                # Make actual prediction
                risk_score = predictor.predict(user_inputs)
                
                if risk_score is not None:
                    st.success("✅ Risk Assessment Complete!")
                    
                    # Display risk visualization
                    risk_fig = create_risk_visualization(risk_score)
                    st.plotly_chart(risk_fig, use_container_width=True)
                    
                    # Risk interpretation
                    risk_level, color, recommendations = interpret_risk_score(risk_score)
                    
                    st.markdown(f"### Risk Level: <span style='color: {color}; font-weight: bold;'>{risk_level}</span>", 
                               unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("### 📋 Recommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {rec}")
                        
                else:
                    st.error("Failed to generate prediction. Please check your inputs and try again.")
            else:
                st.error("Model not loaded. Please ensure all model files are uploaded.")
    
    with col2:
        st.markdown("### ℹ️ About This Tool")
        st.info("""
        **Model Information:**
        - Based on CDC BRFSS 2014 data
        - Feed-forward neural network
        - Trained on 21 health indicators
        - Uses SHAP for explainability
        
        **Important Notes:**
        - This is a screening tool only
        - Not a substitute for medical diagnosis
        - Consult healthcare providers for concerns
        - Based on self-reported survey data
        """)
        
        st.markdown("### 📊 Your Input Summary")
        input_df = pd.DataFrame([user_inputs]).T
        input_df.columns = ['Value']
        st.dataframe(input_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>
    This tool is for educational and informational purposes only. 
    Always consult with qualified healthcare professionals for medical advice.
    <br>
    Model trained on CDC BRFSS 2014 data | Built with Streamlit & PyTorch
    </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()