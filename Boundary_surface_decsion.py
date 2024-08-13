import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification,make_circles,make_moons,make_blobs
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import streamlit as st

st.image("innomatics-logo.webp")
st.title('Boundary Surfaces Visualization')
data = st.sidebar.radio(
        "Type of Data: ",
        ["classification", "circles", "blobs","moons"],
        horizontal=True
    )

if data == 'classification':
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
elif data == 'circles':
    X, y = make_circles(n_samples=100, factor=0.5, noise=0.05)
    
elif data == 'blobs':
    X,y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.0, random_state=42)
    
elif data == 'moons':
    X,y = make_moons(n_samples=100,noise=0.1 ,random_state=42)

classifier_name = st.sidebar.radio('Select Classifier',["KNN"],
        horizontal=True)

if classifier_name == 'KNN':
    n_neighbors = st.sidebar.slider('Number of Neighbors (k)', 1, 15, 3)
    weights = st.sidebar.selectbox('Weight Function', ('uniform', 'distance'))
    algorithm = st.sidebar.selectbox('Algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))
    n_jobs = st.sidebar.number_input('Number of Parallel Jobs (n_jobs)', -1, 10, 1)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs)
    knn.fit(X,y)
    plot_decision_regions(X, y, clf=knn)
    st.pyplot(plt)