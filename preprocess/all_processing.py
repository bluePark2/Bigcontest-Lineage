import pandas as pd
import os
import numpy as np


def make_basic_form(activity):
    df = activity.groupby(['acc_id']).sum()
    features = df.columns
    df = df.drop(features, axis=1)
    return df



def activity_gain(df, activity):
    gain_features = ['playtime', 'npc_kill', 'rich_monster', 'revive', 'exp_recovery', 'fishing']
    activity = activity.groupby('acc_id').sum()[gain_features]
    df = df.join(activity)

    return df
def combat_gain(df, combat):
    gain_features = ['pledge_cnt', 'num_opponent']
    combat = combat.groupby('acc_id').sum()['gain_features']
    df = df.join(combat)

    return df

def daily_act(df, activity):
    print('activity daily started...')
    activity = activity.groupby(['acc_id', 'day']).sum()
    gain_features = ['npc_kill','solo_exp','party_exp','quest_exp', 'rich_monster', 'revive', 'exp_recovery', 'fishing']
    activity_playtime = activity['playtime']
    activity_gain = np.sum(activity[gain_features], axis=1)

    for i in range(4):
        df['act_Q' + str(i + 1)] = 0
        df['playtime_Q' + str(i+1)] = 0
    c = 0
    l = len(df.index)
    #----------process act_gain------------#
    for i in df.index:
        m = activity_gain.loc[i]
        for j in m.index:
            t = int( (j-1)/7)
            df.loc[i,'act_Q'+ str(t+1)] += m.loc[j]
            df.loc[i,'playtime_Q'+str(t+1)] += activity_playtime.loc[(i,j)]
        if c % int(l / 5) == 0:
            print(c / l * 100, "%")
        c = c + 1
    print(df)

    return df
def daily_com(df, combat):
    print('combat daily started...')
    index = np.sort(combat['acc_id'].unique())
    combat = combat.groupby(['acc_id', 'day']).sum()
    gain_features = ['pledge_cnt', 'num_opponent']
    combat_gain = np.sum(combat[gain_features], axis=1)

    for i in range(4):
        df['com_Q' + str(i + 1)] = 0
    c = 0
    l = len(index)
    for i in index:
        m = combat_gain.loc[i]
        for j in m.index:
            t = int((j - 1) / 7)
            df.loc[i, 'com_Q' + str(t + 1)] += m.loc[j]
        if c % int(l / 5) == 0:
            print(c / l * 100, "%")
        c = c + 1
    return df


def add_attribute_to_pledge(df, pledge):
    df['attribute_to_pledge'] = 0
    pledge = pledge.groupby('acc_id').count()

    for i in pledge.index:
        df.loc[i, 'attribute_to_pledge'] = pledge.loc[i, 'day']
    return df

def add_day_feature(df, activity):
    df['days'] = 0
    activity = activity.groupby(['acc_id', 'day']).count()
    activity = activity.groupby('acc_id').count()
    df['days'] = activity['char_id']

    return df


def merchant(df, activity):
    df['merchant'] = 0
    activity = activity.groupby('acc_id').sum()
    merchant = activity[activity['private_shop'] > 0]
    for i in merchant.index:
        df.loc[i, 'merchant'] = merchant.loc[i, 'private_shop']
    return df


def add_most_pledge_id_feature(df, pledge):
    df['most_pledge_id'] = 0
    ids = pledge['acc_id'].unique()

    for i in ids:
        p = pledge[pledge['acc_id'] == i]['pledge_id'].value_counts()
        df.loc[i, 'most_pledge_id'] = p.idxmax()

    ser = df['most_pledge_id'].value_counts()
    max = ser.iloc[0]

    if ser.index[0] == 0:
        ser.iloc[0] = 0
        max = ser.iloc[1]

    df['pledge_size'] = 0

    for i in df.index:
        df.loc[i, 'pledge_size'] = ser[df.loc[i, 'most_pledge_id']] / max

    df.drop('most_pledge_id', axis=1, inplace=True)
    return df


def add_payment(df, payment):
    payment = payment.groupby('acc_id').sum()[['amount_spent']]
    payment.rename(columns={'amount_spent': 'payment'}, inplace=True)
    df = df.join(payment)
    df.fillna(0, inplace=True)
    df['payment'] = df['payment']/28

    return df


def all_preprocessing(activity, combat, pledge, payment):
    df = make_basic_form(activity)
    df = daily_act(df, activity)
    df = daily_com(df, combat)
    df = activity_gain(df, activity)
    df = combat_gain(df, combat)
    df = add_attribute_to_pledge(df, pledge)
    df = add_day_feature(df, activity)
    df = merchant(df, activity)
    df = add_most_pledge_id_feature(df, pledge)
    df = add_payment(df, payment)
    return df


os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')

train_activity = pd.read_csv('train_activity.csv')
train_combat = pd.read_csv('train_combat.csv')
train_pledge = pd.read_csv('train_pledge.csv')
train_payment = pd.read_csv('train_payment.csv')
test1_activity = pd.read_csv('test1_activity.csv')
test1_combat = pd.read_csv('test1_combat.csv')
test1_pledge = pd.read_csv('test1_pledge.csv')
test1_payment = pd.read_csv('test1_payment.csv')
test2_activity = pd.read_csv('test2_activity.csv')
test2_combat = pd.read_csv('test2_combat.csv')
test2_pledge = pd.read_csv('test2_pledge.csv')
test2_payment = pd.read_csv('test2_payment.csv')

df1 = all_preprocessing(train_activity, train_combat, train_pledge, train_payment)
df2 = all_preprocessing(test1_activity, test1_combat, test1_pledge, test1_payment)
df3 = all_preprocessing(test2_activity, test2_combat, test2_pledge, test2_payment)
os.chdir('C:/Users/beomj/PycharmProjects/Lineage')

#df1 = pd.read_csv('train_xdata.csv').set_index('acc_id').drop('payment', axis=1)
#df2 = pd.read_csv('test1_xdata.csv').set_index('acc_id').drop('payment', axis=1)
#df3 = pd.read_csv('test2_xdata.csv').set_index('acc_id').drop('payment', axis=1)

#df1 = add_payment(df1, train_payment)
#df2 = add_payment(df2, test1_payment)
#df3 = add_payment(df3, test2_payment)


#df1 = pd.read_csv('train_xdata.csv').set_index('acc_id')
#df1 = daily_act(df1, train_activity)

df1.to_csv('train_xdata.csv')
df2.to_csv('test1_xdata.csv')
df3.to_csv('test2_xdata.csv')

