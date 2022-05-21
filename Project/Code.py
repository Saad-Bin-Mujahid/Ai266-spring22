#CONTRIBUTION OF SAAD BIN MUJAHID
from google.colab import drive
drive.mount('/content/drive')
import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, KFold
train=panda.read_csv('/content/drive/MyDrive/train.csv')
y = train.target
X = train.drop(['target','f_27'],axis=1)
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print(t_train.shape)
t_train.drop_duplicates(inplace=True)
counts = train.nunique()
to_del = [i for i,v in enumerate(counts) if v == 1]
print(to_del)
t_train.drop(to_del, axis=1, inplace=True)
print(t_train.shape)
t_train.drop_duplicates(inplace=True)
print(t_train.shape)
clf = MultinomialNB(alpha=0)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
NoMNB=clf.score(t_test,y_test)
print("ACCURACCY BY NO SMOOTHING:- ",NoMNB*100)

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

#CONTRIBUTION OF SARIM RAZA
#LAPLACE SMOOTHING
clf = MultinomialNB(alpha=1)
clf.fit(abs(t_train),y_train)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
LPMNB=clf.score(t_test,y_test)
print("ACCURACY BY LAPLACE SMOOTHING:- ",LPMNB*100)
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

#CONTRIBUTION OF MUHAMMAD HUZAIFA
#LIDSTONE SMOOTHING
clf = MultinomialNB(alpha=0.5)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
LDMNB=clf.score(t_test,y_test)
print("ACCURACY BY LIDSTONE SMOOTHING:- ",LDMNB*100)
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))


#CONTRIBUTION OF SAAD BIN MUJAHID & SARIM RAZA 
extra_tree = ExtraTreeClassifier(random_state=0,criterion ='entropy')
cls = BaggingClassifier(extra_tree, random_state=0).fit(t_train, y_train)
cls = BaggingClassifier(extra_tree, random_state=0).fit(t_train, y_train)
Cover_type = cls.predict(t_test)
print(Cover_type)
ETree=cls.score(t_test,y_test)
print(ETree*100)
clf = RidgeClassifier().fit(t_train, y_train)
Cover_type = clf.predict(t_test)
print(,Cover_type) 
Ridge=clf.score(t_test,y_test)
print(,Ridge*100)
clf = RandomForestClassifier(max_depth=100, random_state=0)
clf.fit(t_train,y_train)
Cover_type = clf.predict(t_test)
print(,Cover_type)
RandomForest=clf.score(t_test,y_test)
print(,RandomForest*100)
test=panda.read_csv('/content/drive/MyDrive/test.csv')
test.head()
test = test.drop('f_27', axis=1)
dtree_model = DecisionTreeClassifier(criterion="entropy",max_depth = 20000000).fit(t_train, y_train)
Target = dtree_model.predict(t_test)
print(Target)
Tree=dtree_model.score(t_test,y_test)
print(Tree*100)
Target=dtree_model.predict(test)
print(Target)
sample = test[['id']].copy()
sample['target'] = Target
print(sample)
sample.to_csv('sample.csv',index=False)
clf = MultinomialNB(alpha=0)
clf.fit(abs(t_train),y_train)
target=clf.predict(test)
print("The Predicted Values",target)

#CONTRIBUTION OF SYED TAHA ANWER
#CROSS-VALIDATION
cv = KFold(n_splits=5, random_state=1, shuffle=True)
model = MultinomialNB(alpha=0)
scores = cross_val_score(model, abs(t_train), y_train, scoring='neg_mean_squared_error',cv=cv, n_jobs=-1)
sample = test[['id']].copy()
sample['target'] = target
print(sample)

#SAMPLE FILE
sample.to_csv('sample.csv',index=False)
