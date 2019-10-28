import pandas as pd
import os

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')
pd.set_option('display.max_columns',40)

def add_day_feature(df, activity):

    df['days'] = 0
    activity = activity.groupby(['acc_id','day']).count()
    activity = activity.groupby('acc_id').count()
    df['days'] = activity['char_id']

    return df


activity = pd.read_csv('train_activity.csv')
df = pd.read_csv('myWorkTest.csv').set_index('acc_id', drop=True)
df = add_day_feature(df, activity)
print(df)

