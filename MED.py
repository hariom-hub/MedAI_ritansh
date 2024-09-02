# Correct the unterminated triple-quoted string literal in the script

# Save the corrected code to a file named 'attractive_disease_prediction_app_v4.py'

# Streamlit UI
import streamlit as st

st.set_page_config(page_title="Disease Predictor", layout="wide")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Function to load data and train model
@st.cache_data
def load_data_and_train_model():
    # Load the dataset
    file_path = 'MedAI.csv'
    data = pd.read_csv(file_path, encoding='ascii')

    # Prepare the data
    X = data.drop('Disease', axis=1)
    y = data['Disease']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return X.columns, model


# Load data and train model
symptom_columns, trained_model = load_data_and_train_model()

# Title and introduction
st.title('Interactive Disease Prediction App')
st.markdown("""
This app uses machine learning to predict potential diseases based on symptoms. 
Please note that this is a demonstration and should not be used for actual medical diagnosis.
Always consult with a healthcare professional for medical advice.
""")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header('Symptom Checklist')
    st.write('Please select all applicable symptoms:')

    # Organize symptoms into categories (this is a simplified example)
    symptom_categories = {
        "General": symptom_columns[:10],
        "Respiratory": symptom_columns[10:20],
        "Gastrointestinal": symptom_columns[20:30],
        "Other": symptom_columns[30:]
    }

    # Create input fields for each symptom, organized by category
    symptoms = {}
    for category, category_symptoms in symptom_categories.items():
        st.subheader(category)
        cols = st.columns(3)
        for i, symptom in enumerate(category_symptoms):
            symptoms[symptom] = cols[i % 3].checkbox(symptom)

with col2:
    st.header('Prediction')
    if st.button('Predict Disease', key='predict'):
        # Prepare input data
        input_data = pd.DataFrame([symptoms])

        # Make prediction
        prediction = trained_model.predict(input_data)
        prediction_proba = trained_model.predict_proba(input_data)

        st.success(f'Predicted Disease: **{prediction[0]}**')

        # Display top 3 most likely diseases
        top_3_indices = prediction_proba[0].argsort()[-3:][::-1]
        top_3_diseases = trained_model.classes_[top_3_indices]
        top_3_probabilities = prediction_proba[0][top_3_indices]

        st.write('Top 3 most likely diseases:')
        for disease, prob in zip(top_3_diseases, top_3_probabilities):
            st.write(f'- {disease}: {prob:.2%}')

        st.warning(
            'Remember: This prediction is based on a simplified model and should not be considered as medical advice.')

# Disclaimer
st.markdown('---')
st.caption(
    'Disclaimer: This app is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.')

# Save the corrected code to a file
with open('attractive_disease_prediction_app_v4.py', 'w') as f:
    f.write('''
# Streamlit UI
import streamlit as st
st.set_page_config(page_title="Disease Predictor", layout="wide")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Function to load data and train model
@st.cache_data
def load_data_and_train_model():
    # Load the dataset
    file_path = 'MedAI.csv'
    data = pd.read_csv(file_path, encoding='ascii')

    # Prepare the data
    X = data.drop('Disease', axis=1)
    y = data['Disease']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return X.columns, model

# Load data and train model
symptom_columns, trained_model = load_data_and_train_model()

# Title and introduction
st.title('Interactive Disease Prediction App')
st.markdown("""
This app uses machine learning to predict potential diseases based on symptoms. 
Please note that this is a demonstration and should not be used for actual medical diagnosis.
Always consult with a healthcare professional for medical advice.
""")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header('Symptom Checklist')
    st.write('Please select all applicable symptoms:')

    # Organize symptoms into categories (this is a simplified example)
    symptom_categories = {
        "General": symptom_columns[:10],
        "Respiratory": symptom_columns[10:20],
        "Gastrointestinal": symptom_columns[20:30],
        "Other": symptom_columns[30:]
    }

    # Create input fields for each symptom, organized by category
    symptoms = {}
    for category, category_symptoms in symptom_categories.items():
        st.subheader(category)
        cols = st.columns(3)
        for i, symptom in enumerate(category_symptoms):
            symptoms[symptom] = cols[i % 3].checkbox(symptom)

with col2:
    st.header('Prediction')
    if st.button('Predict Disease', key='predict'):
        # Prepare input data
        input_data = pd.DataFrame([symptoms])

        # Make prediction
        prediction = trained_model.predict(input_data)
        prediction_proba = trained_model.predict_proba(input_data)

        st.success(f'Predicted Disease: **{prediction[0]}**')

        # Display top 3 most likely diseases
        top_3_indices = prediction_proba[0].argsort()[-3:][::-1]
        top_3_diseases = trained_model.classes_[top_3_indices]
        top_3_probabilities = prediction_proba[0][top_3_indices]

        st.write('Top 3 most likely diseases:')
        for disease, prob in zip(top_3_diseases, top_3_probabilities):
            st.write(f'- {disease}: {prob:.2%}')

        st.warning('Remember: This prediction is based on a simplified model and should not be considered as medical advice.')

# Disclaimer
st.markdown('---')
st.caption('Disclaimer: This app is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.')
''')

