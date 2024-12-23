import streamlit as st
import plotly.express as px
import json
import requests


st.title("Histogram based on the model")


model_name = st.selectbox(
    "What model to choose?", ("RandomForest", "LogisticRegression")
)


st.write("Select value of feature")
test_data = st.slider("test_data", 0, 1, 1)

# Формирование данных для запроса
inputs = {"modelName": model_name, "test_data": [[test_data]]}

# Кнопка для отправки запроса
if st.button("Let's check"):
    # Отправка POST-запроса
    response = requests.post(
        url="http://127.0.0.1:8000/predict",
        data=json.dumps(inputs),
    )
    y_pred = response.json()["predictions"]
    if isinstance(y_pred, list) and len(y_pred) > 0:
        # Построение гистограммы
        graph = px.histogram(y_pred, title="Prediction Histogram")
        st.plotly_chart(graph)
    else:
        st.error("Server returned an unexpected response.")
