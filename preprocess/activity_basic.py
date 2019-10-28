import xgboost
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import os
import modifier
import numpy as np
import matplotlib.pyplot as plt
os.chdir('C:/Users/beomj/PycharmProjects/Lineage')
pd.set_option('display.max_columns', 50)
load_path = 'C:/Users/beomj/PycharmProjects/Lineage'
save_path = 'C:/Users/beomj/PycharmProjects/Lineage'
file_trainX = 'train_xdata.csv'
file_trainY = 'train_label.csv'
file_test1 = 'test1_xdata.csv'
file_test2 = 'test2_xdata.csv'
save_name_1 = 'test1_predict.csv'
save_name_2 ='test2_predict.csv'

df = pd.read_csv(file_trainX)
print(df.describe())