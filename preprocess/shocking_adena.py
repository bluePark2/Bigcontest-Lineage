import pandas as pd
import os

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')

activity = pd.read_csv('train_activity.csv').sort_values(['acc_id','day']).reset_index()
df = pd.read_csv('train_xdata.csv').set_index('acc_id')
df['shocking_adena'] = 0
print(activity)

length = len(activity)
for i in activity.index:
    if i< length-1:
        if activity.loc[i,'game_money_change'] < -3:
            if activity.loc[i+1,'day']- activity.loc[i,'day'] >3:
                df.loc[activity.loc[i,'acc_id'], 'shocking_adena'] += -activity.loc[i,'game_money_change']

print(df.corr(method='pearson')[['survival_time', 'amount_spent']])
