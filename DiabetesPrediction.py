#This program is written by M Anas Ramzan, as a project of internship supervised by Technohachs edu tceh
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
dds=pd.read_csv('D://ML Programs//Intern_Projects//Diabetes_prediction//diabetes.csv')
#print(dds.shape)
#print(dds.describe()) #describe the stats of all people having diabetes or not
#print(dds['Outcome'].value_counts())
#print(dds.groupby('Outcome').mean()) # saperating the data w.r.t labels and then finding the mean of data 
X=dds.drop(columns='Outcome', axis=1)
Y=dds['Outcome']
#standardize the random data to make it easy for algorithm to understand
X=StandardScaler().fit_transform(X)
#print(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model=svm.SVC(kernel='linear').fit(X_train,Y_train)
#Reading input from the user
input_str = input("Enter 8 numbers from data saperated by comma i.e:- 4,110,92,0,0,37.6,0.191,30: ")
#Spliting the input string into individual numbers
numbers_str = input_str.split(',')
#Converting the strings to floating-point values
numbers = [float(num_str) for num_str in numbers_str]
# Creating a list containing these numbers
input_list = numbers
#function for model prediction
def predict_with_model(in_data):
    ip_as_np=np.asarray(in_data)
    #resshaping the data
    in_reshape=ip_as_np.reshape(1,-1)
    #standardize the random data to make it easy for algorithm to understand
    st_data=StandardScaler().fit_transform(in_reshape)
    prediction=model.predict(st_data)
    return prediction
# Calling the model function with the input data
prediction_result = predict_with_model(input_list)
print("Input List:", input_list)
if prediction_result==0:
    print("Patient has NO Diabetes")
elif prediction_result==1:
    print("Patient is Diabetic!!!")
print("Model Prediction Result:", prediction_result)

#Deploying machine learning trained model
filename='trained_model_diabetes prediction.sav'
pickle.dump(model,open(filename,'wb')) #dump to save model
loaded_model=pickle.load(open('trained_model_diabetes prediction.sav','rb'))
