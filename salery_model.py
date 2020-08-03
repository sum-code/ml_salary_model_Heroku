# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df=pd.read_csv('hiring.csv')
df['experience'].fillna(0,inplace=True)
df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(),inplace=True)
def convert_int(word):
    dic={'one':1,'two':2,'three':3,'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0,0:0}
    return(dic[word])
df['experience']=df['experience'].apply(lambda x:convert_int(x))
X=df.iloc[:,:3].values

y=df.iloc[:,-1].values

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X,y)

#saving model to disk for further deployment
pickle.dump(regression,open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))