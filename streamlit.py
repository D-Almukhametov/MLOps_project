import streamlit as st
import plotly.express as px
import json
import requests

st.title('Histogram based on the model')

# Выбор модели
modelName = st.selectbox('What model to choose?', ("RandomForest", "LogisticRegression"))

st.write('Select the number of examples to predict')
num_examples = st.number_input('Number of examples', min_value=1, max_value=100, value=4)

# Генерация списка X значений
X = [[i] for i in range(num_examples)]

inputs = {
    "modelName": modelName,
    "X": X  
}

if st.button("Let's check"):
    try:
        response = requests.post(
            url="http://127.0.0.1:8000/predict",
            data=json.dumps(inputs),
            headers={"Content-Type": "application/json"}
        )
        
        st.write("Response status code:", response.status_code)
        st.write("Response text:", response.text)
        
        data = response.json()
        
        if "detail" in data:
            st.error(f"Error from server: {data['detail']}")
        else:
            predictions = data.get("predictions", [])
            st.write("Predictions:", predictions)
            if predictions:
                graph = px.histogram(predictions, nbins=len(set(predictions)),
                                     labels={'value': 'Class'}, 
                                     title="Predictions Distribution",
                                     text_auto=True)
                st.plotly_chart(graph)
            else:
                st.write("No predictions returned.")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON: {e}")

