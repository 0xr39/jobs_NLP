import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Dataset.csv")

# Data Cleaning
dataset = dataset.dropna(inplace=False) 
dataset = dataset.drop(index=(dataset.loc[(dataset['Education']==' N/A')].index))

# Encoding categorical data
dataset = pd.concat([pd.get_dummies(dataset["Education"]), dataset.drop("Education", axis=1)], axis=1)
dataset = dataset.reset_index(drop=True)
# Cleaning the texts
import re 
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,6886):
    review = re.sub('[^0-9a-zA-Z]',' ', dataset['Requirements'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    print(i)

# Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 8000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,:5].values

