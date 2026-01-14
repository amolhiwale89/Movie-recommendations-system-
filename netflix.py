import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from sklearn.neighbors import NearestNeighbors
import streamlit as st


@st.cache_data
def load_data():
    df=pd.read_csv("cleaned_netflix.csv")
    df.fillna("",inplace=True)
    return df


def encode_feature(df):
    text_data=(df['director']+' '+
    df['cast']+' '+
    df['country']+' '+
    df['listed_in']+' '+
    df['description']
    )
    tfidf=TfidfVectorizer(stop_words='english',max_features=5000)
    text_encoded=tfidf.fit_transform(text_data)


    ## one hot encoding for categorical features
    ohe=OneHotEncoder()
    type_encoded=ohe.fit_transform(df[['type']])

    ##scaling for numerical features
    scaler=MinMaxScaler()
    numeric_encoded=scaler.fit_transform(df[['release_year','duration_minutes']])

    ## combining all features
    final_features=hstack([text_encoded,type_encoded,numeric_encoded]).tocsr()

    return final_features

df = load_data()
final_features=encode_feature(df)

## train the knn model
def train_knn(features):
    model=NearestNeighbors(metric='cosine',algorithm='brute')
    model.fit(features)
    return model
knn_model=train_knn(final_features)

## recommendation function
def recommend(title,n=5):
    idx=df[df['title']==title].index[0]
    distances,indices=knn_model.kneighbors(final_features[idx:idx+1],n_neighbors=n+1)

    rec_indices=indices[0][1:]  # Exclude the first one as it is the same movie
    results=df.iloc[rec_indices][['title','listed_in','type']]
    return results

# Streamlit app
st.title("Netflix Movie/TV Show Recommender")
movie=st.selectbox("Select a Movie/TV Show:",df['title'].unique())    
    
if st.button("Recommend"):
    recommendations=recommend(movie)
    st.write("Recommended Movies/TV Shows:")
    st.dataframe(recommendations)

