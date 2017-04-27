import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


def init_data():
    #X_para = [[89.84061],[192.975],[285.27845],[408.62479],[504.46589],[600.3061],[757.24187],[890.9946],[964.84587],[1060.85033],[1188.43961]]
    #Y_para = [104.447,204.42528,293.86674,385.62031,518.73491,610.00451,789.62124,916.57449,983.44609,1106.56541,1211.8552]

    X_para = [
        [0.004073],
        [0.008426],
        [0.012009],
        [0.015224],
        [0.018834],
        [0.022827],
        [0.026118],
        [0.030146],
        [0.046021],
        [0.048527],
        [0.049801],
        [0.051235],
        [0.052587]
    ]
    Y_para = [
        [0.003926],
        [0.008378],
        [0.012167],
        [0.015614],
        [0.018846],
        [0.022438],
        [0.026332],
        [0.03059],
        [0.040193],
        [0.043331],
        [0.044548],
        [0.045764],
        [0.04698]
    ]

    print X_para,'\n',Y_para
    return X_para,Y_para

def linear_model_main(X,Y,predict_value):
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions

def show_linear_line(X_parameters,Y_parameters):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters,Y_parameters,color='blue')
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)

    plt.title('pm25_intake')

    plt.text(0.015, 0.04, r'Y = 0.85692683*X + 0.00219082')
    plt.xlabel('device1')
    plt.ylabel('device2')
    plt.grid(True)
    plt.xticks(())
    plt.yticks(())
    plt.show()


X,Y = init_data()
predictvalue = 700
result = linear_model_main(X,Y,predictvalue)
print "Intercept value " , result['intercept']
print "coefficient" , result['coefficient']
print "Predicted value: ",result['predicted_value']
show_linear_line(X,Y)