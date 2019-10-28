import pandas as pd
import os

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')

def trim_server(df, activity):

    activity = activity.groupby(['acc_id', 'server']).count()
    df['server'] = 0

    for id, server in activity.index:
        if df.loc[id, 'server'] == 0:
            df.loc[id, 'server'] = server

    dict ={}
    count = 1
    for i in df.index:
        if not dict.__contains__(df.loc[i,'server']):
            dict[df.loc[i,'server']] = count
            count += 1
        df.loc[i,'server']= dict[df.loc[i,'server']]

    return df

def trim_level(df, combat):
    df['level'] = 0
    activity = pd.read_csv('train_activity.csv')
    combat = combat[['acc_id', 'level']]
    combat = combat.groupby(['acc_id']).max()

    for i in combat.index:
        df.loc[i, 'level'] = combat.loc[i, 'level']

    return df

def trim_class(df, combat):

    df['class'] = 0

    for i in df.index:
        temp = combat[combat['acc_id']==i]
        level = df.loc[i,'level']
        temp = temp[temp['level'] == level]
        df.loc[i,'class'] = temp['class'].iloc[0]

    return df



df = pd.read_csv('기존df파일.csv').set_index('acc_id')
activity = pd.read_csv('train_activity.csv')
combat = pd.read_csv('train_combat.csv')[['acc_id','level','class']]

df = trim_server(df, activity)
df = trim_level(df, combat)
df = trim_class(df, combat)

df.to_csv('filename.csv')

