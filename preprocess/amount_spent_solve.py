import pandas as pd
import correlation
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')
pd.set_option('display.max_columns',30)

df = pd.read_csv('train_label.csv').set_index('acc_id').sort_index()
trade = pd.read_csv('train_trade.csv')
combat = pd.read_csv('train_combat.csv')
pledge = pd.read_csv('train_pledge.csv')
activity = pd.read_csv('train_activity.csv')
payment = pd.read_csv('train_payment.csv')

drop_feature = ['day','char_id','pledge_id']
t = pledge.drop(drop_feature, axis=1)
t = t.groupby('acc_id').sum()
df = df.join(t).drop('survival_time', axis=1)[['amount_spent', 'play_char_cnt']]
df.fillna(0, inplace=True)
df = df[df['amount_spent']<0.2]


plt.figure()
df1 = df[df['amount_spent']==0]
df2 = df[df['amount_spent']!=0]

print(df1.describe())
print(df2.describe())

plt.plot(df1.describe().loc['mean',:],'bo')
plt.plot(df2.describe().loc['mean',:],'ro')
plt.legend(['low', 'high'])
plt.show()


correlation.correlation(df)
