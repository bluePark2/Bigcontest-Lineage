import pandas as pd
import os


os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')
pd.set_option('display.max_columns', 30)
def add_payment(df, payment):

    sum = payment.groupby('acc_id').sum()
    mean = payment.groupby('acc_id').mean()
    sum=sum.rename(columns = {'amount_spent':'sum of amount_spent'})
    mean=mean.rename(columns = {'amount_spent':'mean of amount_spent'})
    df = df.join(sum[['sum of amount_spent']])
    df = df.join(mean[['mean of amount_spent']])
    df.fillna(0, inplace=True)
    return df

df = pd.read_csv('test2_xdata.csv').set_index('acc_id')
payment = pd.read_csv('test2_payment.csv')
df = add_payment(df, payment)
os.chdir('C:/Users/beomj/PycharmProjects/Lineage')

df.to_csv('test2_xdata.csv')

