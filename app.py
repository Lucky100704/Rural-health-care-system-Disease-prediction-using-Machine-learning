from flask import Flask, render_template, request
import joblib  # For loading the trained machine learning model
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained models
bp_model = joblib.load('bp_model.pkl')  # Load the blood pressure model
diabetes_model = joblib.load('diabetes_model.pkl')  # Load the diabetes model
cold_type_model_and_encoder = joblib.load('cold_type_model_and_encoder.pkl')  # Load the cold prediction model
cold_type_model = cold_type_model_and_encoder['model']
label_encoder = cold_type_model_and_encoder['label_encoder']
asthma_model = joblib.load('asthma_model.pkl')  # Load the asthma prediction model
gastroenteritis_model = joblib.load('gastroenteritis_model.pkl')  # Load the gastroenteritis model

# Blood pressure category descriptions
bp_category_descriptions = {
    'Normal': "Systolic < 120 mmHg, Diastolic < 80 mmHg. This represents a healthy blood pressure level.",
    'Elevated': "Systolic 120–129 mmHg and Diastolic < 80 mmHg. This is an early stage of increased blood pressure, but not yet hypertension.",
    'Hypertension Stage 1': "Systolic 130–139 mmHg or Diastolic 80–89 mmHg. Blood pressure is higher than normal and requires lifestyle changes or medication.",
    'Hypertension Stage 2': "Systolic ≥ 140 mmHg or Diastolic ≥ 90 mmHg. This represents more severe high blood pressure, which may require immediate medical attention."
}

@app.route('/')
def index():
    return render_template('index.html')  # Main page with options for both models

@app.route('/blood_pressure')
def blood_pressure():
    return render_template('blood_pressure.html')  # Blood pressure input page

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')  # Diabetes input page

@app.route('/cold_prediction')
def cold_prediction():
    return render_template('cold_prediction.html')  # Cold prediction input page

@app.route('/asthma')
def asthma():
    return render_template('asthma.html')  # Asthma prediction input page

@app.route('/gastroenteritis')
def gastroenteritis():
    return render_template('gastroenteritis.html')  # Gastroenteritis input page
@app.route('/fever')
def fever():
    return render_template('fever.html')

@app.route('/predict_blood_pressure', methods=['POST'])
def predict_blood_pressure():
    # Get systolic and diastolic values from the form
    systolic = int(request.form['systolic'])
    diastolic = int(request.form['diastolic'])

    # Use the blood pressure model to predict the category
    prediction = bp_model.predict([[systolic, diastolic]])

    # Convert prediction to a more readable format
    if prediction == 'Normal':
        result = 'Normal: Systolic <120 mmHg and Diastolic <80 mmHg — Healthy, no action needed.'
        description = bp_category_descriptions['Normal']
    elif prediction == 'Elevated':
        result = 'Systolic 120-129 mmHg and Diastolic <80 mmHg — Increased risk, lifestyle changes recommended.'
        description = bp_category_descriptions['Elevated']
    elif prediction == 'Hypertension Stage 1':
        result = 'Hypertension Stage 1: Systolic 130-139 mmHg or Diastolic 80-89 mmHg — High risk, lifestyle changes and possibly medication needed.'
        description = bp_category_descriptions['Hypertension Stage 1']
    elif prediction == 'Hypertension Stage 2':
        result = 'Hypertension Stage 2: Systolic ≥140 mmHg or Diastolic ≥90 mmHg — Severe risk, requires medication and close monitoring.'
        description = bp_category_descriptions['Hypertension Stage 2']
    else:
        result = 'Unknown Blood Pressure Category'
        description = 'This category is unknown or an error occurred during classification.'

    return render_template('blood_pressure.html', prediction=result, description=description)

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    # Get input values from the form (Glucose, SkinThickness, Insulin, Age)
    glucose = float(request.form['glucose'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    age = int(request.form['age'])

    # Prepare the features for prediction
    features = np.array([[glucose, skin_thickness, insulin, age]])

    # Use the diabetes model to predict
    prediction = diabetes_model.predict(features)

    result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    return render_template('diabetes.html', prediction=result)

@app.route('/predict_cold_type', methods=['POST'])
def predict_cold_type():
    # Get the symptom values from the form
    fever = int(request.form['fever'])
    cough = int(request.form['cough'])
    sore_throat = int(request.form['sore_throat'])
    runny_nose = int(request.form['runny_nose'])
    headache = int(request.form['headache'])

    # Prepare input data for prediction
    input_data = np.array([[fever, cough, sore_throat, runny_nose, headache]])

    # Predict cold type using the model
    prediction = cold_type_model.predict(input_data)

    # Decode the prediction using the label encoder
    predicted_class = label_encoder.inverse_transform(prediction)

    return render_template('cold_prediction.html', prediction_text=f'The predicted cold type is: {predicted_class[0]}')

@app.route('/predict_asthma', methods=['POST'])
def predict_asthma():
    # Get input values from the form (Age, Gender, Smoking status, Family history, Wheezing, Shortness of breath, Chest tightness)
    age = int(request.form['age'])
    shortness_of_breath = int(request.form['shortness_of_breath'])
    chest_tightness = int(request.form['chest_tightness'])
    coughing = int(request.form['coughing'])
    wheezing = int(request.form['wheezing'])
    family_history = int(request.form['family_history'])
    dust_allergy = int(request.form['dust_allergy'])
    smoking = int(request.form['smoking'])
    cold_air_trigger = int(request.form['cold_air_trigger'])
    exercise_trigger = int(request.form['exercise_trigger'])

    # Prepare the features for prediction
    features = np.array([[age, shortness_of_breath, chest_tightness, coughing,wheezing, family_history, dust_allergy, smoking,cold_air_trigger, exercise_trigger]])

    # Predict asthma using the model
    prediction = asthma_model.predict(features)

    # Display the result
    result = '🫁 Asthma Detected' if prediction == 1 else '✅ No Asthma'
    return render_template('asthma.html', prediction=result)

@app.route('/predict_gastroenteritis', methods=['POST'])
def predict_gastroenteritis():
    # Get input values from the form (fever, vomiting, diarrhea, abdominal pain, bloody diarrhea)
    fever = int(request.form['fever'])
    vomiting = int(request.form['vomiting'])
    diarrhea = int(request.form['diarrhea'])
    abdominal_pain = int(request.form['abdominal_pain'])
    bloody_diarrhea = int(request.form['bloody_diarrhea'])

    # Prepare the input data for prediction
    input_data = np.array([[fever, vomiting, diarrhea, abdominal_pain, bloody_diarrhea]])

    # Predict the type using the gastroenteritis model
    prediction = gastroenteritis_model.predict(input_data)

    # Convert the prediction to a readable format
    if prediction == 0:
        result = 'Viral Gastroenteritis'
    else:
        result = 'Bacterial Gastroenteritis'

    return render_template('gastroenteritis.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
