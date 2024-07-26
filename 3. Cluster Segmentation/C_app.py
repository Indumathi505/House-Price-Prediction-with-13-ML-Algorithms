

import streamlit as st
import numpy as np
import pickle

# Load the pre-trained KMeans model
filename1 = "kmeans_model.pkl"
with open(filename1, "rb") as file:
    loaded_model1 = pickle.load(file)

filename2 = 'hierarchy_model.pkl'
with open(filename2,'rb') as file:
    loaded_model2 = pickle.load(file)

# Custom CSS for colorful representation

st.markdown(
    """
    <style>
    .title {
        color: #FF5733;
        text-align: center;
        font-size: 32px;
    }
    .text {
        color: #7D3C98;
        text-align: center;
        font-size: 18px;
    }
    .prediction {
        color: #6C3483;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<p class="title">Customer Cluster Segmentation</p>', unsafe_allow_html=True)

st.markdown('<p class="text">Enter the annual income,spending score and choose clustering method to predict cluster.</p>', unsafe_allow_html=True)

# Collect user input features
Annual_Income = st.number_input("Annual Income (k$):", min_value=0.0, max_value=150.0, value=50.0)
Spending_Score = st.number_input("Spending Score (1-100):", min_value=1, max_value=100, value=50)
Clustering_Method = st.selectbox("Select the Clustering Method:",("KMeans","Hierarchical"))


# Feature input for prediction
feature = np.array([[Annual_Income, Spending_Score]])

# Predict button
if st.button('Predict Cluster'):    
    if Clustering_Method == "KMeans":
        cluster = loaded_model1.predict(feature)[0]+1
    elif Clustering_Method == "Hierarchical":
        labels = loaded_model2.labels_
        cluster = labels[0]+1
    else:
        cluster = "Select a valid clustering method"
        
    st.markdown(f'<p class="prediction">Predicted cluster:{cluster}<p>',unsafe_allow_html = True)
