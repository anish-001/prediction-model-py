from flask import Flask, request, jsonify

application = Flask(__name__)
application.secret_key='Diabetes'


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix 
  

data_set= pd.read_csv("diabetesfrankfurt.csv") 
  

x= data_set.iloc[:, 0:8].values  
y= data_set.iloc[:, 8].values  



from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

 
from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(x_train, y_train)  

y_pred= classifier.predict(x_test)


test_data_accuracy = accuracy_score(y_pred, y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

@application.route("/predict", methods=['GET','POST'])
def predicform():
    d={}
    input_data = str(request.args['data'])
    lst=input_data.split(',')
    arr=[]
    for i in range (0,8):
        arr.append(float(lst[i]))
    print(arr)
    input_data_as_numpy_array = np.asarray(arr)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    std_data = st_x.transform(input_data_reshaped)

    prediction = classifier.predict(std_data)


    print(prediction)
    d=str(prediction[0])
    return d

if __name__ == '__main__':
    application.run(debug=True)