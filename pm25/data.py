import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

#定义两坐标数据
def init_data():
    X_para = [
        [0.003592],
        [0.007718],
        [0.011411],
        [0.01545],
        [0.017127],
        [0.018804],
        [0.022401],
        [0.025225],
        [0.026776],
        [0.028792],
        [0.031472]
    ]
    Y_para = [
        [0.003238],
        [0.006337],
        [0.009112],
        [0.011135],
        [0.013465],
        [0.015062],
        [0.01906],
        [0.022191],
        [0.023227],
        [0.025136],
        [0.026768]
    ]

    print X_para,'\n',Y_para
    return X_para,Y_para

#线性回归计算
def linear_model_main(X,Y,predict_value):
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions

#回归结果展示
def show_linear_line(X_parameters,Y_parameters):
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters,Y_parameters,color='blue')
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
    #标题
    plt.title('pm25_intake')
    #图中指示内容以及位置
    plt.text(0.009, 0.025, r'Y = 0.88554282*X - 0.000922')
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