#!/usr/bin/env python
# coding: utf-8
# ## Oasis Infobyte
# ### Author- Akshata Naganath Nichal
# ### EMAIL SPAM DETECTION WITH MACHINE LEARNING
# ### Importing Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
### Importing DataSet
data=pd.read_csv("C:/Users/91986/Downloads/spam.csv",encoding="ISO-8859-1", engine="python")
data
data.isnull().sum()
data.shape
data.info()
# ### Data Preprocessing
data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
data.head()
data.rename(columns={'v1':'target','v2':'text'},inplace=True)
data.head()
data.replace({'target':{'ham':0,'spam':1}},inplace=True)
data.head()
data.duplicated().sum()
data.shape
# ### Data Visualisation
data['length']=data.text.apply(len)
data.head()
plt.figure(figsize=(15,6))
sns.lineplot(x=data['target'],y=data['length'],data=data,palette='his')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(10,6))
sns.barplot(x=data['target'],y=data['length'],data=data,palette='hls')
plt.xticks(rotation=90)
plt.show()
ax=plt.subplots(figsize=(10,4))
sns.kdeplot(data.loc[data.target==0,"length"],shade=True,label="Ham",clip=(-50,250))
sns.kdeplot(data.loc[data.target==1,"length"],shade=True,label="Spam")
ax.set(xlabel="Length",ylabel="Density",title="Length of Messeges")
ax.legend(loc="upper right")
plt.show()
# ### Spliting The Data
x=data['text']
x.head()
y=data["target"]
y.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
trainCV=cv.fit_transform(x_train)
testCV=cv.transform(x_test)
# ### Performing Support Vector Machine 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
classifier_svm_linear=SVC(kernel='linear')
classifier_svm_linear.fit(trainCV,y_train)
pred_svm_linear=classifier_svm_linear.predict(testCV)
Accuracy_Score_SVM_Linear=accuracy_score(y_test,pred_svm_linear)
Accuracy_Score_SVM_Linear
print("Support Vector Machine Linear=",Accuracy_Score_SVM_Linear)
# ### Performing Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier_dt=DecisionTreeClassifier()
classifier_dt.fit(trainCV,y_train)
pred_dt=classifier_dt.predict(testCV)
Accuracy_Score_dt=accuracy_score(y_test,pred_dt)
Accuracy_Score_dt
print("Decision Tree Classifier=",Accuracy_Score_dt)
# ### Thank You...
