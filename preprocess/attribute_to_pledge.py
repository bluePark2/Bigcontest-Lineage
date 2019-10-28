import pandas as pd
import os
os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')
pd.set_option('display.max_columns',50)

pledge= pd.read_csv('train_pledge.csv')
df = pd.read_csv('myWorkTest.csv').set_index('acc_id')

def add_attribute_to_pledge(df, pledge):
    df['attribute_to_pledge'] = 0
    pledge = pledge.groupby('acc_id').count()

    for i in pledge.index:
        df.loc[i,'attribute_to_pledge'] =pledge.loc[i,'day']
    return df

