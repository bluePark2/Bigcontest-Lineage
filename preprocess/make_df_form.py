import pandas as pd
import os

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')

def make_basic_form(activity):

    df = activity.groupby(['acc_id']).sum()
    features = df.columns
    df = df.drop(features, axis = 1)
    return df

activity = pd.read_csv('train_activity.csv')
df = make_basic_form(activity)

df.to_csv('train_xdata.csv')
