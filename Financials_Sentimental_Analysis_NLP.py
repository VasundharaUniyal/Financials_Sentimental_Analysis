#!/usr/bin/env python
# coding: utf-8

# ## About Dataset
# Data
# The following data is intended for advancing financial sentiment analysis research. It's two datasets (FiQA, Financial PhraseBank) combined into one easy-to-use CSV file. It provides financial sentences with sentiment labels.
# 
# Citations
# Malo, Pekka, et al. "Good debt or bad debt: Detecting semantic orientations in economic texts." Journal of the Association for Information Science and Technology 65.4 (2014): 782-796.

# In[189]:


from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import nltk
import re
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from collections import Counter, defaultdict

from datasets import Dataset
import torch
import transformers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,precision_score,accuracy_score,f1_score

from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments, DistilBertTokenizerFast,DistilBertForSequenceClassification

import warnings
warnings.filterwarnings('ignore')

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')


# In[11]:


## Attention mechanism models - bert 
## BOW - idf , cdf
## tokenization


# In[12]:





# In[190]:


df = pd.read_csv('Sentimental.csv')
df


# In[191]:


def preprocess_text(text):
    text = str(text).lower() 
    text = text.replace('{html}', "")
    clean = re.compile('<.*?>')
    clean_data = re.sub(clean, '', text)
    remove_url = re.sub(r'http\S+', '',clean_data)
    remove_num = re.sub('[0-9]+', '', remove_url)
    tokenizer =  RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(remove_num) 
    words = [i for i in tokens if len(i)>2 if not i in stopwords.words('english')]
    stem_words=[stemmer.stem(i) for i in words]
    lemma_words=[lemmatizer.lemmatize(i) for i in stem_words]
    return " ".join(words)

df['Cleaned_text'] = df['Sentence'].apply(preprocess_text)
df[['Sentence','Cleaned_text']].head()


# In[192]:


import string
from nltk.tokenize import word_tokenize
from nltk.corpus import words


# In[193]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[194]:


vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['Cleaned_text']).toarray()
print(vectorizer.get_feature_names_out())
y = df['Sentiment']


# In[195]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[196]:


print(f' Training test size:{X_train.shape[0]}') # 80%
print(f' Test set size:{X_test.shape[0]}') #20%


# In[197]:


# Applying logistic regression.


# In[198]:


from sklearn.linear_model import LogisticRegression


# In[199]:


model = LogisticRegression(max_iter = 1000)


# In[200]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[201]:


model.fit(X_train,y_train)


# In[202]:


y_pred = model.predict(X_test)


# In[203]:


y_pred


# In[204]:


print(f'Accuracy: {accuracy_score(y_test,y_pred):.2f}')
print(classification_report(y_test,y_pred))


# In[136]:


import joblib


# In[205]:


from sklearn.svm import LinearSVC


# In[206]:


X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_text'], 
                                                    df['Sentiment'],
                                                    test_size=0.2,
                                                    random_state=0,
        
                                                   )


# In[207]:


X_train.shape


# In[208]:


X_test.shape


# In[209]:


clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])


# In[210]:


clf.fit(X_train, y_train)


# In[211]:


y_pred = clf.predict(X_test)


# In[212]:


y_pred


# In[213]:


print(classification_report(y_test, y_pred))


# In[224]:


joblib.dump(clf,'Sentimental_analysis.pkl')
loaded_model = joblib.load('Sentimental_analysis.pkl')


# In[227]:


loaded_model.predict(['Wow, this is amazing lesson'])


# In[228]:


loaded_model.predict(["i am sad"]) 


# In[ ]:





# In[217]:


import pickle

pickle.dump(clf, open('sentiment_analysis.pkl', 'wb'))


# In[ ]:




