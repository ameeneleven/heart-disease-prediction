from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib  # For saving/loading the model

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session usage

# Load the dataset and train the model
def train_model():
    # Load the dataset from a local file
    df = pd.read_csv('heart_disease.csv', header=None)
    
    # Add column names (based on the UCI Heart Disease dataset)
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df.columns = columns
    
    # Prepare features (X) and target (y)
    X = df.drop(columns='target')
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train RandomForest model
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    
    # Save the model to a file using joblib
    joblib.dump(clf, 'heart_disease_model.pkl')
    return clf

# Check if model already exists, if not train it
try:
    clf = joblib.load('heart_disease_model.pkl')  # Try loading the model if it exists
except:
    clf = train_model()  # Train the model if it's not found

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form and ensure they are numeric
        age = float(request.form['age'])
        sex = int(request.form['sex'])  # 1 for male, 0 for female
        cp = int(request.form['cp'])  # Chest pain type
        trestbps = float(request.form['trestbps'])  # Resting blood pressure
        chol = float(request.form['chol'])  # Serum cholesterol
        fbs = int(request.form['fbs'])  # Fasting blood sugar > 120 mg/dl
        restecg = int(request.form['restecg'])  # Resting electrocardiographic results
        thalach = float(request.form['thalach'])  # Maximum heart rate achieved
        exang = int(request.form['exang'])  # Exercise induced angina
        oldpeak = float(request.form['oldpeak'])  # Depression induced by exercise relative to rest
        slope = int(request.form['slope'])  # Slope of the peak exercise ST segment
        ca = int(request.form['ca'])  # Number of major vessels colored by fluoroscopy
        thal = int(request.form['thal'])  # Thalassemia: 3 = normal; 6 = fixed defect; 7 = reversible defect
        
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                                  columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        # Predict the probability of heart attack
        heart_attack_prob = clf.predict_proba(input_data)[0][1] * 100  # Convert to percentage

        # Store results in session to pass to result page
        session['heart_attack_chance'] = round(heart_attack_prob, 2)
        
        return redirect(url_for('result'))
    except ValueError as e:
        return f"Invalid input. Please ensure all fields are filled correctly. Error: {e}", 400

@app.route('/result')
def result():
    heart_attack_chance = session.get('heart_attack_chance')
    return render_template('result.html', heart_attack_chance=heart_attack_chance)

if __name__ == '__main__':
    app.run(debug=True)
