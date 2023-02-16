# # Oasis Infobyte 
# ### Auther - Akshata Naganath Nichal
# ### PERFOMING MACHINE LEARNING MODEL ON IRIS FLOWER DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
data=pd.read_csv("C:/Users/91986/Downloads/IRIS.csv")
data.head()
data.tail() 
data.isnull().sum() 
data.shape 
data.dtypes 
data['Species'].unique() 
data.describe() 
# # Data Visualization
sns.pairplot(data)
data.corr()
sns.heatmap(data.corr(),annot=True)
plt.show()
plt.boxplot(data['SepalLengthCm'])
plt.show()
#  From above heatmap we can see that there is no outliear in the SepalLengthCm
plt.boxplot(data['SepalWidthCm'])
plt.show()
# from above boxplot we can see that there are some outliear predict in SepalWidth
plt.boxplot(data['PetalLengthCm'])
plt.show()
# from above boxplot we can see that there are some outliear predict in PetalLength
plt.boxplot(data['PetalWidthCm'])
plt.show()
# from above boxplot we can see that there are some outliear predict in PetalWidth
data.drop('Id',axis=1, inplace=True)
spec={'Iris-setosa':1,'Iris-versicolor':2, 'Iris-virginica':3}
data.Species=[spec[i] for i in data.Species]
data
x=data.iloc[:,0:4]
x
y=data.iloc[:,4]
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
# ## Training Model
model=LinearRegression()
model.fit(x,y)
model.score(x,y)  
model.coef_
model.intercept_
# ## Making Predictions
y_pred=model.predict(x_test)
# ## Model Evolation
print("Mean Squared Error: %.2f" % np.mean((y_pred - y_test)**2))
# # Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
accuracies={}
nb = GaussianNB()
nb.fit(x_train, y_train)
acc=nb.score(x_test,y_test)*100
accuracies['Naive Bayes']=acc
print("Accuracy Of Naive Bayes:{:.2f}%".format(acc))
nb.score(x_train,y_train)*100
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
predictions = nb.predict(x_test)
predictions
sns.heatmap(confusion_matrix(y_test, predictions), annot = True)
plt.show()
# # Thank You
