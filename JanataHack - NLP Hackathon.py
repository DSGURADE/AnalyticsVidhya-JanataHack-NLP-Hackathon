#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# ## Reading and Inspection

# In[2]:


# Read the csv file using 'read_csv'
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
game_overview = pd.read_csv('game_overview.csv')


# In[3]:


# Top 5 records of train dataframe
train.head()


# In[4]:


# Check the number of rows and columns in the train dataframe
train.shape


# In[5]:


# Check the column-wise info of the train dataframe
train.info()


# In[6]:


game_overview.head()


# In[7]:


# Check the number of rows and columns in the game_overview dataframe
game_overview.shape


# In[8]:


# Check the column-wise info of the game_overview dataframe
game_overview.info()


# In[9]:


# Check the number of rows and columns in the test dataframe
test.shape


# In[10]:


# Check the column-wise info of the test dataframe
test.info()


# ## Data Cleaning and Data Preparation

# In[11]:


# Get the column-wise Null count

print("Column-wise null count in train dataframe:\n", train.isnull().sum())
print("\n")
print("Column-wise null count in test dataframe:\n", test.isnull().sum())
print("\n")
print("Column-wise null count in game_overview dataframe:\n", game_overview.isnull().sum())


# In[12]:


# Dropping the null values from the dataframe
train.dropna(inplace=True)


# In[13]:


# Changing datatype of column 'year' from float to int
train['year'] = train['year'].apply(np.int64)


# In[14]:


train['year'].value_counts()


# #### Preparing final dataset for training and testing

# In[15]:


# Merging train and test dataframe with game_overview dataframe as train and test df
train = pd.merge(train,game_overview,on='title', how='inner')
train.head()


# In[16]:


# Check the number of rows and columns in the train and test dataframe
print("Size of train dataframe:", train.shape)
print("Size of test dataframe:", test.shape)


# In[17]:


# Get a summary of the train dataframe using 'describe()'
train.describe()


# In[18]:


# Checking for duplicate records is train dataframe
train[train.duplicated()]


# In[19]:


# Checking for duplicate records is test dataframe
test[test.duplicated()]


# #### Text Preprocessing

# In[20]:


# function to remove stopwords from the text
from nltk.corpus import stopwords

# We would like to remove all stop words like a, is, an, the, ... 
# so we collecting all of them from nltk library
stop_words = set(stopwords.words('english'))

def remove_stopwords(col):
    col = col.str.replace("[^\w\s]", "").str.lower()
    # col = col.apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
    return col.apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))  #col.head()


# In[21]:


# function for stemming of text
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def get_stemming_text(col):
    #col = col.str.lower().map(stemmer.stem)
    return col.str.lower().map(stemmer.stem)  #col.head()


# In[22]:


# function for lemmatization of text
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def get_lemmatized_text(col):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in col]


# In[23]:


train['user_review'] = remove_stopwords(train['user_review'])
train['user_review'] = get_stemming_text(train['user_review'])
train['user_review'] = get_lemmatized_text(train['user_review'])


# In[24]:


test['user_review'] = remove_stopwords(test['user_review'])
test['user_review'] = get_stemming_text(test['user_review'])
test['user_review'] = get_lemmatized_text(test['user_review'])


# In[25]:


train.head()


# In[26]:


test.head()


# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(train['user_review'])
y = train['user_suggestion']
test_v = tfidf.transform(test['user_review'])


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[29]:


# XGB Classifier
from xgboost import XGBClassifier

xgb = XGBClassifier( learning_rate =0.1,
 n_estimators=112,
 max_depth=9,
 min_child_weight=5,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.6,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=13,
 reg_lambda=5,
# max_delta_step=1,
 alpha=0,
 base_score=0.5,
 seed=1029)

xgb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = xgb.predict(X_test)  

# Accuracy of XGB model
accuracy_xgb = round(xgb.score(X_train, y_train) * 100, 2)
print("Accuracy score of XGB algorithm is:", accuracy_xgb)


# In[30]:


# print f1 score
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)


# In[31]:


# Predicting the Test set results
test_predicted = xgb.predict(test_v)


# In[32]:


# load loan_id of test dataset
test_review_id = test['review_id']
print(test_review_id.shape)


# In[33]:


# save results to csv
submission_file = pd.DataFrame({'review_id': test_review_id, 'user_suggestion': test_predicted})
submission_file = submission_file[['review_id','user_suggestion']]    
submission_file.to_csv('Final_Solution.csv', index=False)

