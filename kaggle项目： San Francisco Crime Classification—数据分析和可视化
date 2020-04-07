#这是我做的第二项目，不得不感叹只有做项目才能真正的学到东西呀！继续加油！为了将来！
#这项目属于监督学习中多分类问题，可以用logsitic回归，随机森林，贝叶斯，KNN等算法来进行分析。机器学习中数据需要数值型。
#这项目的数据类型：
'''Data columns (total 9 columns):
Dates         878049 non-null object   时间，很重要的指标。需要提取year，month，hour出来，进行数据分析和可视化
Category      878049 non-null object   这是需要预测的目标。一共有39种
Descript      878049 non-null object   数据描述可以去除，test数据中无
DayOfWeek     878049 non-null object   星期，很重要的指标。这也可以用来研究研究，通过分析，周三的犯罪普遍升高
PdDistrict    878049 non-null object   地区，太重要了，这指标
Resolution    878049 non-null object   这个可以去除，test中无
Address       878049 non-null object   上面的地区和这里的地址以及下面的经纬度，会不会有点重复了呢？emmm，确实不是很好拿捏！
X             878049 non-null float64  维度
Y             878049 non-null float64  经度
'''
#数据分析和可视化
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
path=r'....\kaggle\San Francisco Crime Classification\data\train.csv'
Traindata=pd.DataFrame(pd.read_csv(path))
#print(Traindata.head())
#print(Traindata.info())
# 分析Category
Category=Traindata['Category'].value_counts()
#print(Category)

# 各个犯罪所占比例
'''Category_total=0
for i in range(len(Category)):
    Category_total+=Category[i]
def rate(n):
    temp=(n/Category_total)*100
    return temp
ratrio=Category[:].map(rate)
#print(ratrio)
df_rate=pd.DataFrame({'rate':ratrio})
#print(df_rate)
df_rate.plot(kind='bar',title='Rate in Category')
#print('犯罪类型前6所占比例：',df_rate[:6].sum(),'%')
#print('犯罪类型前15所占比例：',df_rate[:15].sum(),'%')
#plt.show()
#Traindata['Category'].value_counts().plot(kind='bar',title='Category')
#plt.show()'''

# 分析DayofWeek
'''print(Traindata['DayOfWeek'].value_counts())
Traindata['DayOfWeek'].value_counts().plot(kind='pie',title='DayOfWeek')
plt.show()'''


# 提取年月日
# 将object类型转为datetime类型
# 用datetime直接提取年月日，高级！
Traindata['date']=pd.to_datetime(Traindata['Dates'])
Traindata['year'] = Traindata.date.dt.year
Traindata['month'] = Traindata.date.dt.month
Traindata['day'] = Traindata.date.dt.day
Traindata['hour'] = Traindata.date.dt.hour


# 分析years
# 用split()函数提取年份
'''def DataTitle(name):
    str=name.split('-')[0]
    return str
Traindata['years']=Traindata['Dates'].map(DataTitle)
years=Traindata['years'].value_counts()
years=years.sort_index()
y=years.values
x=sorted(years.index[:])
plt.plot(x,y,marker='.')
plt.title('Years')
plt.ylabel('Numbers')
plt.show()
'''
'''
# months
def month(name):
    str=name.split('-')[1]
    return str
Traindata['months']=Traindata['Dates'].map(month)
months=Traindata['months'].value_counts()
months=months.sort_index()
y=months.values
x=sorted(months.index[:])
plt.title('Months')
plt.ylabel('Numbers')
plt.plot(x,y,marker='*')
plt.show()
'''
'''
#days
def day(name):
    str=name.split('-')[2]
    str1 = str.split(' ')[0]
    return str1
Traindata['days']=Traindata['Dates'].map(day)
days=Traindata['days'].value_counts()
days=days.sort_index()
y=days.values
x=days.index[:]
#plt.xticks(range(len(x)), x)
plt.plot(x,y,marker='*')
plt.title('Days')
plt.ylabel('Numbers')
#plt.show()
'''

# hours
def hour(name):
    str=name.split(' ')[1]
    str1 = str.split(':')[0]
    return str1
Traindata['hours']=Traindata['Dates'].map(hour)

'''hours=Traindata['hours'].value_counts()
hour1=hours.sort_index()
hour1.plot(marker='*',figsize=(12,8))
plt.title('Hours')
plt.ylabel('Numbers')
#plt.show()
'''

# hours and weeks
'''week_group = Traindata.groupby(['DayOfWeek','hours']).size()#多重分组
week_group = week_group.unstack()#对分组后的多重索引转为xy索引
week_group.T.plot(figsize=(12,8))#行列互换后画图
plt.xlabel('hour of day',size=15)
plt.ylabel('Number of crimes',size=15)
plt.show()
'''

# hours and top6 Crimes
'''top6=Category.index[:6]
print(top6)
temp1=Traindata.hours[(Traindata.Category=='LARCENY/THEFT')].value_counts()#, 'OTHER OFFENSES', 'NON-CRIMINAL', 'ASSAULT', 'DRUG/NARCOTIC', 'VEHICLE THEFT'
temp2=Traindata.hours[(Traindata.Category=='OTHER OFFENSES')].value_counts()
temp3=Traindata.hours[(Traindata.Category=='NON-CRIMINAL')].value_counts()
temp4=Traindata.hours[(Traindata.Category=='ASSAULT')].value_counts()
temp5=Traindata.hours[(Traindata.Category=='DRUG/NARCOTIC')].value_counts()
temp6=Traindata.hours[(Traindata.Category=='VEHICLE THEFT')].value_counts()
df_crimesAndhour=pd.DataFrame({'LARCENY/THEFT':temp1,'OTHER OFFENSES':temp2,'NON-CRIMINAL':temp3,'ASSAULT':temp4,'DRUG/NARCOTIC':temp5,'VEHICLE THEFT':temp6})
df_crimesAndhour.plot(figsize=(12,8))
plt.xlabel('hour of day',size=10)
plt.ylabel('Number of crimes',size=10)
plt.show()
'''
# hours and top6 Crimes2
'''top6=list(Category.index[:6].map(string.capwords))
# 这功能强的离谱，提取满足前6犯罪的数据
tmp = Traindata[Traindata['Category'].map(string.capwords).isin(top6)]
tmp_group=tmp.groupby(['Category','hours']).size()
tmp_group=tmp_group.unstack()
tmp_group.T.plot(figsize=(12,8))
plt.ylabel('hour')
plt.xlabel('numbers')
plt.title('Category and Hours')
plt.show()'''
#描述经纬度情况
#print(Traindata[['X','Y']].describe())

# 分析地区
'''print(Traindata['PdDistrict'].value_counts())
Traindata['PdDistrict'].value_counts().plot(kind='bar',figsize=(12,8))
plt.title('Crimes and district')
plt.xlabel('district')
plt.ylabel('numbers')
plt.show()
'''

# to6Crimes and District
'''top6=list(Category.index[:6].map(string.capwords))
# 这功能强的离谱，提取满足前6犯罪的数据
tmp = Traindata[Traindata['Category'].map(string.capwords).isin(top6)]
tmp_group2=tmp.groupby(['Category','PdDistrict']).size()
tmp_group2=tmp_group2.unstack()
tmp_group2.plot(kind='bar',figsize=(12,8))
plt.title('Category and District')
plt.show()
'''
# District and Hours
'''Traindata['date']=pd.to_datetime(Traindata['Dates'])
Traindata['hour'] = Traindata.date.dt.hour
df_DistrictandHours=pd.DataFrame({'PdDistrict':Traindata['PdDistrict'],'hours':Traindata['hour']})
tmp_group3=Traindata.groupby(['PdDistrict','hours']).size()
tmp_group3=tmp_group3.unstack()
tmp_group3.T.plot(figsize=(12,8))
plt.show()'''

# top6Crimes and weeks
'''top6=list(Category.index[:6].map(string.capwords))
tmp = Traindata[Traindata['Category'].map(string.capwords).isin(top6)]
tmp_groups4=tmp.groupby(['Category','DayOfWeek']).size()
tmp_groups4=tmp_groups4.unstack()
#print(tmp_groups4.sum(axis=1)[0]) #对0行求和
#print(tmp_groups4.iloc[0]) #提取0行的所有列
for i in range(6):
    tmp_groups4.iloc[i] = tmp_groups4.iloc[i]/tmp_groups4.sum(axis=1)[i]
wkm = {
    'Monday':0,
    'Tuesday':1,
    'Wednesday':2,
    'Thursday':3,
    'Friday':4,
    'Saturday':5,
    'Sunday':6
}
tmp_groups4.columns = tmp_groups4.columns.map(wkm)
#print(df_tmap_group4)
# 排序
tmp_groups4=tmp_groups4.sort_index(axis=1)
tmp_groups4.T.plot(figsize=(12,6),style='o-')
plt.xlabel("weekday",size=20)
#plt.axes.set_xticks([])
plt.xticks([0,1,2,3,4,5,6],['Mon','Tue','Wed','Thur','Fri','Sat','Sun'])
plt.show()'''

# top6Crimes and months
'''top6=list(Category.index[:6].map(string.capwords))
tmp = Traindata[Traindata['Category'].map(string.capwords).isin(top6)]
tmp=tmp.groupby(['Category','month']).size()
tmp=tmp.unstack()
for i in range(6):
    tmp.iloc[i]=tmp.iloc[i]/tmp.sum(axis=1)[i]
tmp.T.plot(figsize=(12,8),style='o-')
plt.show()
'''
#top6Crimes and months&years
'''top6=list(Category.index[:6].map(string.capwords))
tmp = Traindata[Traindata['Category'].map(string.capwords).isin(top6)]
tmp=tmp.groupby(['Category','date']).size()
tmp=tmp.unstack().fillna(0) #如果有空值，则用0填充
tmp=tmp.T
tmp.index
tmp2=tmp.resample('m',how='sum')
plt.style.use('ggplot')
moav = tmp2.rolling(12).mean()#每12个月统计平均，相当于加了个窗
i = 1
for cat in tmp2.columns:
    fig=plt.figure(figsize=(12,15))
    ax = fig.add_subplot(6,1,i)
    ax.plot(tmp2.index,tmp2[cat])
    ax.plot(tmp2.index,moav[cat])
    plt.title(cat)
    i+=1
tmp2.plot()
plt.show()
