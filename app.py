from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from pymongo import MongoClient

filename = 'random_forest_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

client = MongoClient('mongodb+srv://saneer2001:jaishreeram@mongopject.bh69gat.mongodb.net/')
db = client['contact_form_db']
collection = db['contact_form']

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/tool')
def tool():
      return render_template('tool.html')

@app.route('/about')
def about_us():
    return render_template('about_us.html')

@app.route('/faqs')
def faqs():
    return render_template('faqs.html')

@app.route('/help')
def help():
     return render_template('help.html')

@app.route('/contacts')
def contacts():
     return render_template('contacts.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Insert form data into MongoDB
        submission = {'name': name, 'email': email, 'message': message}
        collection.insert_one(submission)
        
        return 'Form submitted successfully!'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            sex = request.form.get('sex')
            cp = request.form.get('cp')
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = request.form.get('fbs')
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = request.form.get('exang')
            oldpeak = float(request.form['oldpeak'])
            slope = request.form.get('slope')
            ca = int(request.form['ca'])
            thal = request.form.get('thal')
            
            # Proceed with prediction
            data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
            my_prediction = model.predict(data)
            
            return render_template('result.html', prediction=my_prediction)
        except ValueError:
            return render_template('error.html')
    else:
        # Handle GET request (direct access without form submission)
        return render_template('error.html')
        

if __name__ == '__main__':
	app.run(debug=True)

