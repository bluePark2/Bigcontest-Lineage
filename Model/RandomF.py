import sklearn.preprocessing
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import xgboost

os.chdir('C:/Users/beomj/PycharmProjects/Lineage')
model = RandomForestRegressor()
model2 = xgboost.XGBRegressor()
train_X = pd.read_csv('train_xdata.csv').set_index('acc_id')
train_Y = pd.read_csv('train_label.csv').set_index('acc_id').sort_values('acc_id')


file = 'test1_xdata.csv'
string = 'survival_time'
test = pd.read_csv(file).set_index('acc_id')
index = test.index
output = pd.DataFrame(index=index)
for i in range(2):
    train_y = train_Y[string]
    model2.fit(train_X, train_y)
    predict2 = model2.predict(test)
    output[string] = predict2
    string = 'amount_spent'


print(output)

