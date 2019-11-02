from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import os
import modifier
import numpy as np

load_path = 'C:/Users/beomj/PycharmProjects/Lineage'
save_path = 'C:/Users/beomj/PycharmProjects/Lineage'
file_trainX = 'train_xdata.csv'
file_trainY = 'train_label.csv'
file_test1 = 'test1_xdata.csv'
file_test2 = 'test2_xdata.csv'
save_name_1 = 'test1_predict.csv'
save_name_2 ='test2_predict.csv'

os.chdir(load_path)

train_X = pd.read_csv(file_trainX).set_index('acc_id')
train_Y = pd.read_csv(file_trainY).set_index('acc_id').sort_values('acc_id')
payment_location = train_X.columns.get_loc('payment')

train_Y_s = train_Y['survival_time']
train_X_a = train_X[train_X['payment']<0.75]
train_Y_a = train_Y.loc[train_X_a.index,'amount_spent']

test1 = pd.read_csv(file_test1).set_index('acc_id')
test2 = pd.read_csv(file_test2).set_index('acc_id')
index1 = test1.index
index2 = test2.index

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
train_X_a  = my_imputer.fit_transform(train_X_a)
test1 = my_imputer.transform(test1)
test2 = my_imputer.transform(test2)

rb_scaler1 = RobustScaler()
rb_scaler2 = RobustScaler()

train_X = rb_scaler1.fit_transform(train_X)
train_X_a = rb_scaler2.fit_transform((train_X_a))
test1 = rb_scaler1.transform(test1)
test2 = rb_scaler1.transform(test2)


my_model = RandomForestRegressor()
my_model2 = RandomForestRegressor()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

my_model = RandomizedSearchCV(estimator=my_model, param_distributions=random_grid, n_iter=5, cv=3, verbose=2, \
                                   random_state=42, n_jobs=-1)

my_model2 = RandomizedSearchCV(estimator=my_model2, param_distributions=random_grid, n_iter=5, cv=3, verbose=2, \
                                  random_state=42, n_jobs=-1)



my_model.fit(train_X, train_Y_s)
my_model2.fit(train_X_a, train_Y_a)

predict1_survival = my_model.predict(test1)
predict1_spent= my_model2.predict(test1)
predict = np.c_[predict1_survival, predict1_spent]


predict2_survival = my_model.predict(test2)
predict2_spent= my_model2.predict(test2)

output1, output2 = pd.DataFrame( columns = ['acc_id','survival_time', 'amount_spent']),pd.DataFrame( columns = ['acc_id','survival_time', 'amount_spent'])
output1['acc_id'], output2['acc_id'] = index1, index2
output1['survival_time'],output2['survival_time'] = predict1_survival, predict2_survival
output1['amount_spent'], output2['amount_spent'] = predict1_spent, predict2_spent

os.chdir(save_path)

output1.to_csv(save_name_1, index=False)
output2.to_csv(save_name_2, index=False)

modifier.modifier()


print(my_model.get_params())

