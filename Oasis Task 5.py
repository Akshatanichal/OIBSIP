# # Oasis Infobyte
# ## Task No. 5
# ## SALES PREDICTION USING PYTHON
# ### Author: Akshata Naganath Nichal
# ### Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# ### Importing Dataset
data=pd.read_csv("C:/Users/91986/Downloads/Advertising.csv")
data.sample(5)
data.tail()
data.shape
data.describe()
data.isnull().sum()
data.info()
# ### Data Visualisation
data.corr()
sns.heatmap(data.corr(),cbar=True,linewidths=0.5,annot=True)
sns.pairplot(data)
sns.distplot(data['Newspaper'])
sns.distplot(data['Radio'])
sns.distplot(data['Sales'])
sns.distplot(data['TV'])
# ### Data Preprosasing
data=data.drop(columns=['Unnamed: 0'])
data
x=data.drop(['Sales'],1)
x.head()
y=data['Sales']
y.head()
# ### Spliting the Dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)
print(x.shape,x_train.shape,x_test.shape)
print(y.shape,y_train.shape,y_test.shape)
x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)
from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()
x_train_scaled=Sc.fit_transform(x_train)
x_test_scaled=Sc.fit_transform(x_test)
# ### Applying Linear Regression
from sklearn.linear_model import LinearRegression
accuracies={}
lr=LinearRegression()
lr.fit(x_train,y_train)
acc=lr.score(x_test,y_test)*100
accuracies['Linear Regression']=acc
print("Test Accuracy {:.2f}%".format(acc))
# ### Analyzing the data by Scatter plot
y_pred=lr.predict(x_test_scaled)
plt.scatter(y_test,y_pred,c='r')
### Thank You
