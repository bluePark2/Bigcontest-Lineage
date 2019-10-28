import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')
pd.set_option('display.max_columns',20)



def merchant(df, activity):

    df['merchant'] = 0
    activity = activity.groupby('acc_id').sum()
    merchant = activity[activity['private_shop'] > 0]
    for i in merchant.index:
        df.loc[i,'merchant'] =merchant.loc[i,'private_shop']

    return df

activity = pd.read_csv('train_activity.csv')
df = pd.read_csv('myWorkTest.csv').set_index('acc_id')
df = merchant(df,activity)
print(df.corr(method='pearson'))
