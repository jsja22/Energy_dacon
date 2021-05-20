#Library Imports
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from workalendar.asia import SouthKorea

train=pd.read_csv('C:/data/energy_dacon/energy/train.csv',encoding='cp949')
test=pd.read_csv('C:/data/energy_dacon/energy/test.csv',encoding='cp949')
submission=pd.read_csv('C:/data/energy_dacon/energy/sample_submission.csv',encoding='cp949')

train.columns = ['num', 'date_time', '전력사용량', '기온', '풍속', '습도', '강수량', '일조', '비전기냉방설비운영', '태양광보유']
test.columns = ['num', 'date_time', '기온', '풍속', '습도', '강수량', '일조', '비전기냉방설비운영', '태양광보유']


holidays = pd.Series(np.array(SouthKorea().holidays(2020))[:, 0])
print(holidays) #총 15일
#1.1, 1.24, 1.25, 1.26, 3.1, 4.30, 5.5, 6.6, 8.15, 9.30, 10.01, 10.02, 10.03, 10.09, 12.25

#건물별로 '비전기냉방설비운영'과 '태양광보유'를 판단해 test set의 결측치를 보간해줍니다
train[['num', '비전기냉방설비운영','태양광보유']]
ice={}
hot={}
count=0
print(len(train)) #122400 24시간씩 총 x 60개 건물 x 85일 ( 2020.06.01~2020.08.24)

for i in range(0, len(train), len(train)//60):
    count +=1
    ice[count]=train.loc[i,'비전기냉방설비운영']
    hot[count]=train.loc[i,'태양광보유']

for i in range(len(test)):
    test.loc[i, '비전기냉방설비운영']=ice[test['num'][i]]
    test.loc[i, '태양광보유']=hot[test['num'][i]]

print(ice)
print(hot)
print(test.head())

#시간 변수와 요일 변수를 추가해봅니다.
def time(x):
    return int(x[-2:])

def weekday(x):
    return pd.to_datetime(x[:10]).weekday()

def month(x):
    return pd.to_datetime(x[:10]).month

def week(x):
    return pd.to_datetime(x[:10]).week

def day(x):
    return pd.to_datetime(x[:10]).day

def holiday(x):
    return int(pd.to_datetime(x[:10]).date().strftime("%Y-%m-%d") in holidays)

def weekend(x):
    if x == 0 or x == 1 or x == 2 or x == 3:
        return 0
    else:
        return 1

def date(x):
    return pd.to_datetime(x[:10]).date().strftime("%Y-%m-%d")

def tomorrow(x):
    return (pd.to_datetime(x[:10]).date() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

train['time']=train['date_time'].apply(lambda x: time(x))
test['time']=test['date_time'].apply(lambda x: time(x))
train['weekday']=train['date_time'].apply(lambda x :weekday(x))
test['weekday']=test['date_time'].apply(lambda x :weekday(x))
train['month']=train['date_time'].apply(lambda x :month(x))
test['month']=test['date_time'].apply(lambda x :month(x))
train['week']=train['date_time'].apply(lambda x :week(x))
test['week']=test['date_time'].apply(lambda x :week(x))
train['day']=train['date_time'].apply(lambda x :day(x))
test['day']=test['date_time'].apply(lambda x :day(x))
train['holiday']=train['date_time'].apply(lambda x :holiday(x))
test['holiday']=test['date_time'].apply(lambda x :holiday(x))
train['weekend']=train['weekday'].apply(lambda x :weekend(x))
test['weekend']=test['weekday'].apply(lambda x :weekend(x))
# train['date']=train['date_time'].apply(lambda x :date(x))
# test['date']=test['date_time'].apply(lambda x :date(x))
# train['tomorrow']=train['date_time'].apply(lambda x :tomorrow(x))
# test['tomorrow']=test['date_time'].apply(lambda x :tomorrow(x))

test.interpolate(method='values')

#print(test.head())
#print(train.head())
#train.to_csv('C:/data/energy_dacon/energy/train2.csv', index=False)
#test.to_csv('C:/data/energy_dacon/energy/test2.csv', index=False)

train_x=train.drop('전력사용량', axis=1)
train_y=train[['전력사용량']]

train_x.drop('date_time', axis=1, inplace=True)
test.drop('date_time', axis=1, inplace=True)

cross=KFold(n_splits=5, shuffle=True, random_state=42)
folds=[]
for train_idx, valid_idx in cross.split(train_x, train_y):
    folds.append((train_idx, valid_idx))

# from sklearn.metrics import f1_score

def SMAPE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred))) #100은 상수이므로 이번 코드에서는 제외

models={}
for fold in range(5):
    print(f'===================={fold+1}=======================')
    train_idx, valid_idx=folds[fold]
    X_train=train_x.iloc[train_idx, :]
    y_train=train_y.iloc[train_idx, :]
    X_valid=train_x.iloc[valid_idx, :]
    y_valid=train_y.iloc[valid_idx, :]
    
    # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], 
    #          early_stopping_rounds=30, verbose=100, eval_metric='mae')
    # models[fold]=model
    model=xgb.XGBRegressor(n_estimators=4000,learning_rate=0.08,gamma=0, subsample=0.75, colsample_bytree=1,max_depth=7)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], 
             early_stopping_rounds=30, verbose=100, eval_metric='mae')
    models[fold]=model
    print(f'================================================\n\n')
for i in range(5):
    submission['answer'] += models[i].predict(test)/5 


#성능 평가
from sklearn.metrics import mean_squared_error,r2_score
y_pred =model.predict(X_train)
mse_score = mean_squared_error(y_train,y_pred)
r2_score =r2_score(y_train,y_pred)
print('mse_score:', mse_score)
print("r2_score:", r2_score)


# n_estimator 500
# mse_score: 26867.65741512175
# r2_score: 0.9936302267322278

#n_estimator 4000
# mse_score: 2912.7804773489847
# r2_score: 0.9993094391917822


#제출
submission.to_csv('C:/data/energy_dacon/energy/submission/baseline_submission2.csv', index=False)


