import requests
import streamlit as st

st.title("Welcome to iris model! 🌸")

try:
    requests.get("http://backend:8000/healthcheck")

    with st.form("Iris parameters"):
        sepal_width = st.number_input(label="Sepal width", min_value=0, max_value=10, value=5)
        sepal_height = st.number_input(label="Sepal height", min_value=0, max_value=10, value=5)
        petal_width = st.number_input(label="Petal width", min_value=0, max_value=10, value=5)
        petal_height = st.number_input(label="Petal height", min_value=0, max_value=10, value=5)

        submitted = st.form_submit_button("Predict!")

        if submitted:
            response = requests.get(
                "http://backend:8000/predict",
                json={
                    "sepal_width": sepal_width,
                    "sepal_height": sepal_height,
                    "petal_width": petal_width,
                    "petal_height": petal_height,
                }
            )
            iris_type = response.json()["predict"]

            st.write(f"This iris is likely {iris_type}.")
except:
    st.warning("BE is unreachable.")
