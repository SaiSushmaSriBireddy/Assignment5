#!/usr/bin/env python
# coding: utf-8

# In[5]:


#importing set of libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics


# In[11]:


#importing the given dataset glass.csv
dsetgiven_Data = pd.read_csv("glass.csv")
dsetgiven_Data.info()


# In[12]:


#splitting the dataset which is excluding last columns
X = dsetgiven_Data.iloc[:, :-1]
y = dsetgiven_Data.iloc[:, -1]
#splitting the dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#creating a Gaussian Naive Bayes model
gn = GaussianNB()
#fitting train data
gn.fit(X_train, y_train)
# predicting the test dataset
y_pred = gn.predict(X_test)
# evaluating the model on the test dataset
print("Accuracy: ", accuracy_score(y_test, y_pred)*100)
print("Classification Report: \n", classification_report(y_test, y_pred))


# In[8]:


#importing set of libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# In[13]:


#loading the glass dataset
dsetgiven_Data = pd.read_csv("glass.csv")
dsetgiven_Data.info()


# In[14]:


#splitting the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#creating a linear SVM model
svm = SVC(kernel='linear')
#fitting the training dataset
svm.fit(X_train, y_train)
#predicting the target values using the test dataset
y_pred = svm.predict(X_test)
#evaluating the model on the test dataset
print("Accuracy: ", accuracy_score(y_test, y_pred)*100)
print("Classification Report: \n", classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




