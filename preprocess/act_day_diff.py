import pandas as pd
import os
os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')
pd.set_option('display.max_columns',50)



def add_act_day_diff_feature(df, activity):

    df['act_day_diff'] = 0
    activity = activity.groupby(['acc_id', 'day']).count()
    activity = activity.reset_index()

    for i in df.index:
        days = list(activity[activity['acc_id'] ==i]['day'])
        days.append(28)
        days.insert(0,0)
        if len(days) ==1:
            df.loc[i, 'act_day_diff'] = 0
        else:
            max_diff = 0
            temp = days[0]
            for j in range(len(days)-1):
                if days[j+1] - temp > max_diff:
                    max_diff = days[j+1] - temp
                temp = days[j+1]
            df.loc[i,'act_day_diff'] = 28 - max_diff
    return df


activity = pd.read_csv('train_activity.csv')
df = pd.read_csv('train_xdata.csv').set_index('acc_id', drop=True)
df = add_act_day_diff_feature(df, activity)

df.to_csv('train_xdata.csv')