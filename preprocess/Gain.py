import pandas as pd
import os

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')
combat = pd.read_csv('train_combat.csv')
pd.set_option('display.max_columns', 20)

def combat_gain_loss(df, combat):
    print('combat preprocessing is started!')
 # Make Columns day12~ day2728
    data = combat
    for i in range(14):
        string =  'D'+str((2*i+1)) + str(2*i+2) +'CombatGain'
        df[string] = 0
        string2 = 'D'+str((2*i+1)) + str(2*i+2) +'CombatLoss'
        df[string2] = 0

 # Gain Features

    gain_features = ['pledge_cnt', 'random_attacker_cnt', 'same_pledge_cnt']
    loss_features = ['random_defender_cnt', 'temp_cnt', 'num_opponent']
    data = data.groupby(['acc_id','day']).sum()

    data['gain'] = 0
    data['loss'] = 0

# Add gain, loss features to df
    print('Adding gain and loss features...')
    print(' * It will take some minutes')
    for i in gain_features:
        data['gain'] += data[i]
    for j in loss_features:
        data['loss'] += data[j]
    data = data[['gain','loss']]

    count = 0
    length = len(data)
    for i,j in data.index:
        k=j
        if j%2 !=0:
            k = j+1
        string = 'D'+str(k-1) +str(k) +'CombatGain'
        string2 = 'D' + str(k - 1) + str(k)+'CombatLoss'
        df.loc[i,string] += data.loc[(i,j),'gain']
        df.loc[i,string2] += data.loc[(i,j),'loss']

        if count%(int(length/20)+1) ==0:
            print(int(count /length *100),'%...')
        count +=1
#data Processing is Done.
    return df


df = combat_gain_loss(df, combat)
print(data)