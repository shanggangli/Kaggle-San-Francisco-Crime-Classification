from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(NewTraindata,target,random_state=0,train_size=0.8)
print(",训练数据特征:",X_train.shape,
      ",测试数据特征:",X_test.shape)
print(",训练数据标签:",y_train.shape,
     ',测试数据标签:',y_test.shape )


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import time
from sklearn.metrics import accuracy_score

#logistic回归
LR=LogisticRegression(multi_class='ovr', penalty='l2', solver='lbfgs', tol=0.01)
LR_starttime=time.time()
LR.fit(X_train,y_train)
LR_costtime=time.time()-LR_starttime
print('花费时间：',LR_costtime)
LR_predict=LR.predict(X_test)
#print(LR.score(y_test,LR_predict))
print(accuracy_score(y_test,LR_predict))

'''one_hot = OneHotEncoder(sparse=False)
y_test = one_hot.fit_transform(y_test)
LR_predict= one_hot.fit_transform(LR_predict)
loss=log_loss(y_test.reshape(1,-1),LR_predict.reshape(1,-1))
print('逻辑回归log损失:',loss)'''

from sklearn.naive_bayes import BernoulliNB #贝叶斯
# 朴素贝叶斯
'''NB=BernoulliNB()
NB_starttime=time.time()
NB.fit(X_train,y_train)
NB_costtime=time.time()-NB_starttime
NB_predict=NB.predict(X_test)
print(accuracy_score(y_test,NB_predict))
#print('朴素贝叶斯log损失值:',log_loss(y_test,NB_predict))
#print('朴素贝叶斯花费时间:',NB_costtime)'''

'''from sklearn.ensemble import RandomForestClassifier #随机森林
# 随机森林
RF=RandomForestClassifier(max_depth=15,n_estimators=300)
RF_start=time.time()
RF.fit(X_train,y_train)
RF_costtime=time.time()-RF_start
RF_predict=RF.predict(X_test)
print('随机森林花费时间:',RF_costtime)
print(accuracy_score(y_test,RF_predict))'''
