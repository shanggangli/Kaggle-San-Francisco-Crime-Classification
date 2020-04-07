
### 数据处理！！！ 将类别数据转化为数值型，
#类别特征：Dates,Descript,DayOfWeek,PdDistrict,Resolution,Address
#数值型特征：X,Y,year,month,day,hour
#时间特征：date
#针对类别特征，其中是对数值大小没有影响，只是单纯的分类，直接用pandas中 get_dummies()，将数据进行分0-1型 如：PdDistrict
#如果在数据中有一些类别特征对数值大小有影响，直接把它们变为数值就好 如：DayOfWeek

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
path=r'...kaggle\San Francisco Crime Classification\data\train.csv'
Traindata=pd.DataFrame(pd.read_csv(path))
# 提取年月日
# 将object类型转为datetime类型
Traindata['date']=pd.to_datetime(Traindata['Dates'])
Traindata['year'] = Traindata.date.dt.year
Traindata['month'] = Traindata.date.dt.month
Traindata['day'] = Traindata.date.dt.day
Traindata['hour'] = Traindata.date.dt.hour

#对分类目标做标签化处理
label = preprocessing.LabelEncoder()
target = label.fit_transform(Traindata.Category)

#去除多余的数据
Traindata.drop(['hours','Descript','Resolution','date','Dates','Category'],axis=1, inplace=True)
#Traindata['year_begin'] = Traindata['month'].apply(lambda x:1 if x==2 else 0)
#Traindata['year_mid']=Traindata['month'].apply(lambda x: 1 if x>=6 and x<=9 else 0)
#Traindata['year_end']=Traindata['month'].apply(lambda x: 1 if x==12 else 0)
#Traindata['morning'] = Traindata['hour'].apply(lambda x:1 if x>=2 and x<=7 else 0)
def getMonthZn(month):
    if(month < 3 or month >= 12): return 1; #冬
    if(month >= 3 and month < 6): return 2; #春
    if(month >= 6 and month < 9): return 3; #夏
    if(month >= 9 and month < 12): return 4; #秋
Traindata['month']=Traindata['month'].map(getMonthZn)

def getHourZn(hour):
    if(hour >= 1 and hour < 8): return 1;
    if(hour >= 8 and hour < 12): return 2;
    if(hour >= 12 and hour < 13): return 3;
    if(hour >= 13 and hour < 15): return 4;
    if(hour >= 15 and hour < 17): return 5;
    if(hour >= 17 and hour < 19): return 6;
    if(hour < 1 or hour >= 19): return 7;
Traindata['hour']=Traindata['hour'].map(getHourZn)

weekdays = {'Monday':0., 'Tuesday':1., 'Wednesday':2., 'Thursday': 3., 'Friday':4.,
'Saturday':5., 'Sunday':6.}
weekends = {'Monday':0., 'Tuesday':0., 'Wednesday':0., 'Thursday': 0., 'Friday':0.,
'Saturday':1., 'Sunday':1.}
Traindata['Weekend']=Traindata['DayOfWeek'].map(weekends)
Traindata['weekday']=Traindata['DayOfWeek'].map(weekdays)

#dummies_hour=pd.get_dummies(Traindata['hour'],prefix='hour')
#dummies_week=pd.get_dummies(Traindata['DayOfWeek'],prefix='DayofWeek')
#dummies_mouth=pd.get_dummies(Traindata['month'],prefix='month')
#dummies_year=pd.get_dummies(Traindata['year'],prefix='year')
dummies_PdDistrict=pd.get_dummies(Traindata['PdDistrict'],prefix='PdDistrict')
Traindata.drop(['PdDistrict','Address','DayOfWeek'],axis=1, inplace=True)
NewTraindata=pd.concat([Traindata,dummies_PdDistrict],axis=1)
#print(NewTraindata.info())
#print(NewTraindata.shape)
