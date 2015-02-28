#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

np.set_printoptions(threshold=np.nan)

def transform(df):
    i = 0
    for timestamp in df['datetime']:
        i += 1
        date_object = datetime.strptime(timestamp.split()[0], '%Y-%m-%d')
        time = timestamp.split()[1][:2]
        date = datetime.date(date_object).weekday()
        df.loc[i-1, 'date'] = date
        df.loc[i-1, 'time'] = time
    return df

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
train, test = transform(df_train), transform(df_test)


cols = ['date','time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
rf = RandomForestRegressor(n_estimators=200)
casual = rf.fit(train[cols], train.casual)
registered = rf.fit(train[cols], train.registered)

print casual.feature_importances_
print registered.feature_importances_

count = train.casual + train.registered

df_submission = pd.DataFrame(test.datetime, count)
pd.DataFrame.to_csv(df_submission ,'randomforest_predict.csv')
