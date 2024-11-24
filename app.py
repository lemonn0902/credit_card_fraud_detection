from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
import smtplib
from email.mime.text import MIMEText
import pandas as pd

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('templates/creditcard.csv')

# Calculate the minimum and maximum times
min_time = df['Time'].min()
max_time = df['Time'].max()

logistic_model = joblib.load('trained_models/logistic_model.pkl')

def send_email(to_address, subject, message):
    from_address = 'amulyavm.cy23@rvce.edu.in'
    password = 'Vinathi@123'
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = from_address
    msg['To'] = to_address
    
    server = smtplib.SMTP('smtp.example.com', 587)
    server.starttls()
    server.login(from_address, password)
    server.sendmail(from_address, to_address, msg.as_string())
    server.quit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        time = float(data['time'])
        amount = float(data['amount'])

        if time < 0 or amount < 0:
            return jsonify({'message': 'Invalid input: Time and amount must be non-negative.'})

        # Preprocess the input similar to your training data
        amount = RobustScaler().fit_transform(np.array(amount).reshape(-1, 1)).flatten()[0]
        time = (time - min_time) / (max_time - min_time)  # Scale time feature correctly

        # Combine into the input format expected by the model
        input_data = np.array([time, amount]).reshape(1, -1)
        
        # Predict
        try:
            prediction = logistic_model.predict(input_data)[0]
        except Exception as e:
            return jsonify({'message': 'Error occurred during prediction: {}'.format(str(e))})

        if prediction == 1:
            send_email('eshithachowdary.nettem@gmail.com', 'Fraud Alert', 'A fraudulent transaction was detected.')
            return jsonify({'message': 'Fraudulent transaction detected.'})
        else:
            return jsonify({'message': 'Normal transaction.'})
    except (ValueError, KeyError):
        return jsonify({'message': 'Invalid data: Please ensure that time and amount are valid numbers.'})

if __name__ == '__main__':
    app.run(debug=True)