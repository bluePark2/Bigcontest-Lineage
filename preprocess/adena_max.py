import pandas as pd
import os

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')
pd.set_option('display.max_columns',20)

activity = pd.read_csv('train_activity.csv')
df = pd.read_csv('myWorkTest.csv').set_index('acc_id')
activity = activity.groupby('acc_id').min()

df= df.join(activity[['game_money_change']])
print(df.head())
print(df.corr(method='pearson'))

