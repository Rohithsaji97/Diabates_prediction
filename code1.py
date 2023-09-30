#import all the required libraries
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#change the directory to where you want to save and load data
os.chdir("R:/Diabetes")

#read data 
data=pd.read_csv("R:/Diabetes/diabetes.csv")

#split the data into result and factors
X=data.drop('Outcome',axis=1)
Y=data['Outcome']

#split data into train and test datasets and their result datasets  
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

#model is created
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#model is compiled
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#data set is put into the model for training
model.fit(X_train, Y_train, epochs=25, batch_size=32, validation_split=0.2)

#using the test datasets the predictions are made and accuracy of the trained model is measured
predictions=model.predict(X_test)
predictions = (predictions > 0.5).astype(int)
accuracy=accuracy_score(predictions, Y_test)
print(accuracy)

#the model is saved for using it later
#model.save('DT.h5')
