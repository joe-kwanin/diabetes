import numpy as np
import pandas as pd
from flask_cors import  cross_origin
from flask import Flask, request, render_template
import pickle
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array




# loads the imgage classification mode
model = load_model('diabetes_model.h5')
print(" * Model loaded!")

#Preprocess the image by setting it to the right size for the model to classify
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

app = Flask(__name__)

#flask decorator that links the method to the html page
@cross_origin()
@app.route('/')
def home():
    return render_template('index.html')
@cross_origin()
@app.route('/predict',methods=['GET','POST'])
def predict():
    print(request.form) # this prints to the cmd the entered values by the user
    # these values are then assigned to these variables
    preg = float(request.form['Pregnancies'])
    glucose = float(request.form['Glucose'])
    b_pressure = float(request.form['BloodPressure'])
    skin_thick = float(request.form['SkinThickness'])
    insulin = float(request.form['Insulin'])
    bmi = float(request.form['BMI'])
    dbf = float(request.form['DiabetesPedigreeFunction'])
    age = float(request.form['Age'])
    # the values are converted to a python dictionary and then pandas data  
    df = pd.DataFrame.from_dict({'Pregnancies': [preg],'Glucose':[glucose], 'Blood Pressure':[b_pressure],
                               'Skin Thickness':[skin_thick],'Insulin':[insulin], 
                               'BMI':[bmi], 'Diabetes Pedigree Function':[dbf],'Age': [age]})
    # load the decision tree classifier model from disk
    clf = pickle.load(open("clf_model.sav", 'rb'))
    # Pass the pandas data to classifier clf for prediction
    pred = clf.predict_proba(df)[0]
    # the results is stored as a dictionary and printed in the cmd
    results = {'Non-Diabetic': pred[0], 'Diabetic': pred[1]}
    print(results)
    
    # this retrieves the tongue image uploaded by the user and assigns it to a variable
    tongue = request.files['pic']
    img = tongue.read()

    image = Image.open(io.BytesIO(img))
    # the image is preprocessed to the right size for the model
    processed_image = preprocess_image(image, target_size=(200, 200))

    img_3d=processed_image.reshape(-1,200,200,3) 
    
    t_prediction=model.predict(img_3d).tolist()
    #class_names =['diabetic', 'non diabetic']
    #results2 = {class_names[i]: float(prediction[i]) for i in range(2)}
    
    print(t_prediction[0][1])
    
    # this is the results of the prediction by the decion tree classifier
    if results['Non-Diabetic'] < results['Diabetic']:
            prediction = "diabetes"

    else:
            prediction = "Normal"

    # this is the results from the tongue image classification
    if t_prediction[0][1]==0:
            tongue_predict = "diabetes"

    else:
            tongue_predict = "Normal"

    # showing the prediction results in a UI
    if  prediction =="diabetes":

        return render_template('diabetes.html', tongue_predict=tongue_predict)
    else:
        return render_template('Normal.html', tongue_predict=tongue_predict)

 

if __name__ == "__main__":
    app.run(debug=True)
	#app.run(debug=True)