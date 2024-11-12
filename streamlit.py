import streamlit as st
import plotly.express as px
import json
import requests

st.title('Histogram based on the model')
modelName = st.selectbox('What model to choose?', ("RandomForest", "LogisticRegression"))
st.write('Select value of feature')
X = st.slider('X', 0, 1, 1)
assert modelName == 'RandomForest'
inputs = {
    "modelName": "RandomForest",
    "X": [[0]]
    }
if st.button("Let's check"):
    y_pred = requests.post(url="http://127.0.0.1:8000/predict", data=json.dumps(inputs))
    graph = px.histogram(y_pred)
    st.plotly_chart(graph)
