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
        day = datetime.date(date_object).weekday()
        year_dict = {2011:1, 2012:2}
        year = year_dict[date_object.year]
        df.loc[i-1, 'day'] = day
        df.loc[i-1, 'time'] = time
        df.loc[i-1, 'year'] = year
    return df


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
train, test = transform(df_train), transform(df_test)

cols = ['day', 'time','year', 'season', 'holiday', 'workingday', 'weather','temp', 'atemp', 'humidity', 'windspeed']
rf = RandomForestRegressor(n_estimators=1000, min_samples_split=6, oob_score=True)

casual = rf.fit(train[cols], train.casual)
print casual.feature_importances_
predict_casual = rf.predict(test[cols])

registered = rf.fit(train[cols], train.registered)
print registered.feature_importances_
predict_registered = rf.predict(test[cols])

count = [int(round(i+j)) for i,j in zip(predict_casual, predict_registered)]

df_submission = pd.DataFrame(count, test['datetime'], columns = ['count'])
pd.DataFrame.to_csv(df_submission ,'randomforest_predict.csv')
