#required libraries are inported
from tensorflow.keras.models import load_model
import os
import numpy as np


#the location is changed to where we want to save and load data
os.chdir("R:/Diabetes")

#saved model is called
model= load_model('DT.h5')

#function for collection new input datas where the input datas are returned as lists
def get_data():
    L=[]
    p=int(input('Pregnancies: '))
    L.append(p)
    g=int(input('Glucose: '))
    L.append(g)
    bp=int(input('Blood Pressure: '))
    L.append(bp)
    st=int(input('Skin Thickness: '))
    L.append(st)
    i=int(input('Insuline: '))
    L.append(i)
    bmi=float(input('BMI: '))
    L.append(bmi)
    dpf=float(input('Diabetes Pedigree Function: '))
    L.append(dpf)
    a=int(input('Age: '))
    L.append(a)
    return L


s='Y'
while (s=='Y' or s=='y'):
    #function for collecting data is called
    L=get_data()
    
    #converted to an array for inputing into the model
    input_data=np.array([L])
    
    #preditions are made
    input_predict=model.predict(input_data)
    input_predict = (input_predict > 0.5).astype(int)
    input_predict=input_predict[0]
    input_predict=input_predict[0]
    if input_predict==1:
        print('POSITIVE')
    else:
        print('NEGATIVE')
    s=input("Enter 'Y' to check next patients: ")

