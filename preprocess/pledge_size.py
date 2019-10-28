import pandas as pd
import os

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')
pd.set_option('display.max_columns',16)

pledge = pd.read_csv('train_pledge.csv').sort_values('acc_id')


def add_most_pledge_id_feature(df, pledge):
    df['most_pledge_id'] = 0
    ids = pledge['acc_id'].unique()

    for i in ids:
        p = pledge[pledge['acc_id']==i]['pledge_id'].value_counts()
        df.loc[i,'most_pledge_id'] = p.idxmax()

    ser = df['most_pledge_id'].value_counts()
    max = ser.iloc[0]

    if ser.index[0] ==0:
        ser.iloc[0]=0
        max = ser.iloc[1]

    df['pledge_size'] = 0

    for i in df.index:
        df.loc[i, 'pledge_size'] = ser[df.loc[i,'most_pledge_id']]/max
    return df

df = pd.read_csv('myWorkTest.csv').set_index('acc_id')
df = add_most_pledge_id_feature(df, pledge)
df.to_csv('myWorkTest.csv')

