# CONTRIBUTION OF SAAD BIN MUJAHID (10360)
from google.colab import drive
drive.mount('/content/drive') 
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

train_df=panda.read_csv('/content/drive/MyDrive/train.csv')
y = train_df.target
X = train_df.drop(['target','f_27'],axis=1)
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print(t_train.shape)
t_train.drop_duplicates(inplace=True)
counts = train_df.nunique()
to_del = [i for i,v in enumerate(counts) if v == 1]
print(to_del)
t_train.drop(to_del, axis=1, inplace=True)
print(t_train.shape)
t_train.drop_duplicates(inplace=True)
print(t_train.shape)

clf = MultinomialNB(alpha=0)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
NoMNB=clf.score(t_test,y_test)
print("The Accuracy Of No Smoothing:-",NoMNB*100)
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

# CONTRIBUTION OF SARIM RAZA (11311)
#Laplace Smoothing
clf = MultinomialNB(alpha=1)
clf.fit(abs(t_train),y_train)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
LPMNB=clf.score(t_test,y_test)
print("The Accuracy Of Laplace Smoothing:-",LPMNB*100)
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

# CONTRIBUTION OF MUHAMMAD HUZAIFA (10562)
#Lidstone  Smoothing
clf = MultinomialNB(alpha=0.5)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
LDMNB=clf.score(t_test,y_test)
print("The Accuracy Of Lidstone:-",LDMNB*100)
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

 # CONTRIBUTION OF SYED SALMAN KHURSHID (11260)
#Perceptron
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(abs(t_train), y_train)
clf.predict(t_test)
Percept=clf.score(t_test,y_test)
print("The Accuracy Of Preceptron:-",Percept*100)

# CONTRIBUTION OF SYED TAHA ANWER (10384)
#Support Vector Machine (SVM)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(abs(t_train), y_train)
clf.predict(t_test)
Svm=clf.score(t_test,y_test)
print("The Accuracy Score Of SVM",Svm*100)
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

# CONTRIBUTION OF SAAD BIN MUJAHID (10360)
#K NEAREST NEIGHBOUR (KNN)
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(abs(t_train), y_train)
clf.predict(t_test)
Knn=clf.score(t_test,y_test)
print("The Accuracy Score Of Knn",Knn*100)
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

# CONTRIBUTION OF SAAD BIN MUJAHID (10360)
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))
test=panda.read_csv('/content/drive/MyDrive/test.csv')
test.head()
test = test.drop('f_27', axis=1)
clf = MultinomialNB(alpha=0)
clf.fit(abs(t_train),y_train)
target=clf.predict(test)
print("The Predicted Values",target)
cv = KFold(n_splits=5, random_state=1, shuffle=True) 
model = MultinomialNB(alpha=0)
scores = cross_val_score(model, abs(t_train), y_train, scoring='neg_mean_squared_error',cv=cv, n_jobs=-1)
sample = test[['id']].copy()
sample['target'] = target
print(sample)
sample.to_csv('sample.csv',index=False)
