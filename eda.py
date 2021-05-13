import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
import plotly.express as px

pd.options.plotting.backend = 'plotly'

## plotly.io를 import 한 후 renderers 기본값을 꼭 "notebook_connected" 로 설정해주시기 바랍니다.
import plotly.io as pio
pio.renderers.default = "notebook_connected"

path = 'energy/'
files = sorted(glob(path+'*.csv'))

train = pd.read_csv(files[2], header=0) 
test = pd.read_csv(files[1], header=0)
sample_submission = pd.read_csv(files[0], header=0) 

print(train.head())
print(test.info())
print(test.describe())

fig = px.bar(x=test.columns, y=test.isnull().sum(), title='Null values')
fig.show()

train['date'] = train['date_time'].apply(lambda x: x.split()[0])
train['date_time'] = train['date_time'].apply(lambda x: x.split()[1])
# train['date_time'] = train['date_time'].str.rjust(8,'0') # 한자릿수 시간 앞에 0 추가 ex) 3시 -> 03시

# 24시를 00시로 바꿔주기
train.loc[train['date_time']=='24:00:00','date_time'] = '00:00:00'
train['date_time'] = train['date'] + ' ' + train['date_time']
train['date_time'] = pd.to_datetime(train['date_time'])
train.loc[train['date_time'].dt.hour==0,'date_time'] += timedelta(days=1)

train['month'] = train['date_time'].dt.month
train['hour'] = train['date_time'].dt.hour

mean_month = train.groupby('month').mean()
fig = px.bar(mean_month, x=mean_month.index, y=['전력사용량(kWh)'])
fig.show()

mean_month = train.groupby('month').mean()
fig = px.bar(mean_month, x=mean_month.index, y=['기온(°C)'])
fig.show()

mean_month = train.groupby('month').mean()
fig = px.bar(mean_month, x=mean_month.index, y=['풍속(m/s)'])
fig.show()

mean_month = train.groupby('month').mean()
fig = px.bar(mean_month, x=mean_month.index, y=['습도(%)'])
fig.show()

mean_month = train.groupby('month').mean()
fig = px.bar(mean_month, x=mean_month.index, y=['강수량(mm)'])
fig.show()

mean_month = train.groupby('month').mean()
fig = px.bar(mean_month, x=mean_month.index, y=['일조(hr)'])
fig.show()
## 시간별 발전량
mean_hour = train.groupby('hour').mean()
fig = px.bar(mean_hour, x=mean_hour.index, y=['전력사용량(kWh)'])
fig.show()

mean_hour = train.groupby('hour').mean()
fig = px.bar(mean_hour, x=mean_hour.index, y=['기온(°C)'])
fig.show()

mean_hour = train.groupby('hour').mean()
fig = px.bar(mean_hour, x=mean_hour.index, y=['풍속(m/s)'])
fig.show()

mean_hour = train.groupby('hour').mean()
fig = px.bar(mean_hour, x=mean_hour.index, y=['습도(%)'])
fig.show()

mean_hour = train.groupby('hour').mean()
fig = px.bar(mean_hour, x=mean_hour.index, y=['강수량(mm)'])
fig.show()

mean_hour = train.groupby('hour').mean()
fig = px.bar(mean_hour, x=mean_hour.index, y=['일조(hr)'])
fig.show()

fig = px.imshow(train.corr())
fig.show()

fig = px.imshow(test.corr())
fig.show()
#submission

